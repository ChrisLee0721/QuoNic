"""Backend constant-factor profiles for parametric scheduling.

Each backend/method pair is modeled as:
    T = startup_ms + per_gate_us * gate_count * f(n) / 1000

where f(n) depends on the simulation method's complexity class:
- statevector, density_matrix: 2^n
- stabilizer: n^2
- matrix_product_state: n^2 (approximation)

Profiles are stored in ~/.quonic/profiles.json and updated via EMA
from real execution timings.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .capabilities import eligible_methods
from .registry import BackendRegistry


@dataclass
class BackendProfile:
    """Constant-factor cost model for one backend/method pair."""

    backend_method: str  # e.g. "qpanda/statevector"
    startup_ms: float  # fixed overhead per invocation
    per_gate_us: float  # marginal cost per gate * f(n)
    scaling: str  # "2^n", "n^2"
    n_samples: int = 0  # data points used for fitting
    r_squared: float = 0.0  # goodness of fit
    ema_alpha: float = 0.3  # EMA smoothing factor


@dataclass
class ProfileSet:
    """Collection of backend profiles for scheduling."""

    version: int = 1
    updated_at: str = ""
    profiles: dict[str, BackendProfile] = field(default_factory=dict)


def default_profiles() -> ProfileRegistry | None:
    """Load profiles from ~/.quonic/profiles.json if it exists."""
    path = os.path.join(os.path.expanduser("~"), ".quonic", "profiles.json")
    if os.path.exists(path):
        return ProfileRegistry(path)
    return None


_SCALING_FUNCTIONS = {
    "2^n": lambda n: 2**n,
    "n^2": lambda n: n**2,
    "4^n": lambda n: 4**n,
}


def _scaling_factor(scaling: str, n: int) -> float:
    fn = _SCALING_FUNCTIONS.get(scaling)
    if fn is not None:
        return fn(n)
    return 2**n  # default to statevector scaling


class ProfileRegistry(BackendRegistry):
    """Profile-based scheduling: predict T for each eligible backend/method,
    pick the one with lowest predicted T."""

    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(
            os.path.expanduser("~"), ".quonic", "profiles.json"
        )
        self.profiles: ProfileSet | None = None
        self._update_counter = 0
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            profiles = {}
            for k, v in data.get("profiles", {}).items():
                profiles[k] = BackendProfile(**v)
            self.profiles = ProfileSet(
                version=data.get("version", 1),
                updated_at=data.get("updated_at", ""),
                profiles=profiles,
            )
        except (OSError, ValueError, TypeError):
            self.profiles = None

    def _save(self):
        if self.profiles is None:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            "version": self.profiles.version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "profiles": {k: asdict(v) for k, v in self.profiles.profiles.items()},
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def predict_time(self, backend_method: str, n: int, gate_count: int) -> float | None:
        """Predict execution time using the profile cost model."""
        if self.profiles is None:
            return None
        profile = self.profiles.profiles.get(backend_method)
        if profile is None:
            return None
        f_n = _scaling_factor(profile.scaling, n)
        return profile.startup_ms / 1000 + profile.per_gate_us * gate_count * f_n / 1e6

    def get_recommendation(self, features: dict[str, Any]) -> str | None:
        """Pick the backend/method with lowest predicted time among eligible methods."""
        if self.profiles is None or not self.profiles.profiles:
            return None

        gate_types = features.get("gate_types", [])
        n = features["n"]
        gate_count = features.get("gate_count", n)
        eligible = eligible_methods(gate_types) if gate_types else None

        best_key = None
        best_time = float("inf")

        for bm_key, profile in self.profiles.profiles.items():
            method = bm_key.split("/", 1)[1] if "/" in bm_key else bm_key
            if eligible is not None and method not in eligible:
                continue
            t = self.predict_time(bm_key, n, gate_count)
            if t is not None and t < best_time:
                best_time = t
                best_key = bm_key

        return best_key

    def report_result(
        self,
        features: dict[str, Any],
        backend_name: str,
        duration: float,
        memory: Any | None,
    ) -> None:
        """Update profile via EMA when real timing data arrives."""
        if self.profiles is None:
            return

        with self._lock:
            profile = self.profiles.profiles.get(backend_name)
            if profile is None:
                return

            n = features["n"]
            gate_count = features.get("gate_count", n)
            f_n = _scaling_factor(profile.scaling, n)
            predicted = profile.startup_ms / 1000 + profile.per_gate_us * gate_count * f_n / 1e6

            if predicted > 0:
                ratio = max(0.5, min(2.0, duration / predicted))
                smooth = profile.ema_alpha * ratio + (1 - profile.ema_alpha) * 1.0
                profile.startup_ms *= smooth
                profile.per_gate_us *= smooth

            profile.n_samples += 1
            self._update_counter += 1
            if self._update_counter >= 10:
                self._save()
                self._update_counter = 0


def fit_profiles_from_exp13(checkpoint_path: str) -> ProfileSet:
    """Fit constant-factor profiles from exp13 checkpoint data.

    For each backend/method, collects (n, gate_count, time) tuples and fits
    T = startup + c * gate_count * f(n) via least-squares regression.
    """
    import numpy as np

    with open(checkpoint_path, encoding="utf-8") as f:
        data = json.load(f)

    # Collect data points per backend/method
    # bm_key -> list of (n, gate_count, time)
    observations: dict[str, list[tuple[int, int, float]]] = {}

    for r in data.get("results", []):
        if "all_timings" not in r:
            continue
        n = r["n_qubits"]
        gate_count = r.get("gate_count", n)
        for bm_key, timing_data in r["all_timings"].items():
            t = timing_data["time"] if isinstance(timing_data, dict) else timing_data
            if t <= 0:
                continue
            observations.setdefault(bm_key, []).append((n, gate_count, t))

    profiles: dict[str, BackendProfile] = {}

    for bm_key, points in observations.items():
        if len(points) < 3:
            continue

        method = bm_key.split("/", 1)[1] if "/" in bm_key else bm_key
        scaling = {
            "statevector": "2^n",
            "density_matrix": "4^n",
            "stabilizer": "n^2",
            "matrix_product_state": "n^2",
        }.get(method, "2^n")

        # Build design matrix: T = a + b * gate_count * f(n)
        X_vals = []
        T_vals = []
        for n, gc, t in points:
            f_n = _scaling_factor(scaling, n)
            X_vals.append(gc * f_n)
            T_vals.append(t)

        X_arr = np.array(X_vals)
        T_arr = np.array(T_vals)

        # Filter out extreme values (likely timeouts or errors)
        mask = T_arr < 600  # skip > 10 min
        X_arr = X_arr[mask]
        T_arr = T_arr[mask]

        if len(X_arr) < 3:
            continue

        # Linear regression: T = a + b * X
        A = np.column_stack([np.ones(len(X_arr)), X_arr])
        try:
            result, residuals, rank, sv = np.linalg.lstsq(A, T_arr, rcond=None)
        except np.linalg.LinAlgError:
            continue

        startup_s, per_gate_s_per_unit = result

        # Convert to ms and us
        startup_ms = max(0.0, startup_s * 1000)
        per_gate_us = max(0.0, per_gate_s_per_unit * 1e6)

        # Compute R^2
        T_pred = A @ result
        ss_res = np.sum((T_arr - T_pred) ** 2)
        ss_tot = np.sum((T_arr - np.mean(T_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        profiles[bm_key] = BackendProfile(
            backend_method=bm_key,
            startup_ms=startup_ms,
            per_gate_us=per_gate_us,
            scaling=scaling,
            n_samples=len(X_arr),
            r_squared=max(0.0, r_squared),
        )

    return ProfileSet(
        version=1,
        updated_at=datetime.now(timezone.utc).isoformat(),
        profiles=profiles,
    )


def fit_profiles_from_benchmarks(benchmarks: dict) -> ProfileSet:
    """Fit profiles from benchmarks.json performance data."""
    import numpy as np

    performance = benchmarks.get("performance", [])

    # Collect data points per backend/method
    observations: dict[str, list[tuple[int, int, float]]] = {}

    for entry in performance:
        n = entry["n"]
        gate_count = entry.get("gate_count", n)
        for bm_key, t in entry.get("timings", {}).items():
            if t <= 0:
                continue
            observations.setdefault(bm_key, []).append((n, gate_count, t))

    profiles: dict[str, BackendProfile] = {}

    for bm_key, points in observations.items():
        if len(points) < 3:
            continue

        method = bm_key.split("/", 1)[1] if "/" in bm_key else bm_key
        scaling = {
            "statevector": "2^n",
            "density_matrix": "4^n",
            "stabilizer": "n^2",
            "matrix_product_state": "n^2",
        }.get(method, "2^n")

        X_vals = []
        T_vals = []
        for n, gc, t in points:
            f_n = _scaling_factor(scaling, n)
            X_vals.append(gc * f_n)
            T_vals.append(t)

        X_arr = np.array(X_vals)
        T_arr = np.array(T_vals)

        mask = T_arr < 600
        X_arr = X_arr[mask]
        T_arr = T_arr[mask]

        if len(X_arr) < 3:
            continue

        A = np.column_stack([np.ones(len(X_arr)), X_arr])
        try:
            result, residuals, rank, sv = np.linalg.lstsq(A, T_arr, rcond=None)
        except np.linalg.LinAlgError:
            continue

        startup_s, per_gate_s_per_unit = result
        startup_ms = max(0.0, startup_s * 1000)
        per_gate_us = max(0.0, per_gate_s_per_unit * 1e6)

        T_pred = A @ result
        ss_res = np.sum((T_arr - T_pred) ** 2)
        ss_tot = np.sum((T_arr - np.mean(T_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        profiles[bm_key] = BackendProfile(
            backend_method=bm_key,
            startup_ms=startup_ms,
            per_gate_us=per_gate_us,
            scaling=scaling,
            n_samples=len(X_arr),
            r_squared=max(0.0, r_squared),
        )

    return ProfileSet(
        version=1,
        updated_at=datetime.now(timezone.utc).isoformat(),
        profiles=profiles,
    )
