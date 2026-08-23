"""Backend setup guide: declarative setup + a generic interactive guide engine.

Connecting a real-hardware backend (Quantum Inspire, etc.) requires one-time setup:
install the SDK, log in, and handle version conflicts. These "setup burdens" share the
same structure, so they are abstracted into:

  - Backend.setup: declarative description (dependency / auth / conflicts / devices / billing)
  - diagnose(setup): pure-function diagnostics returning a Diagnosis (unit-testable)
  - ensure_ready(setup): pass when ready; otherwise interactive guide under a TTY, raise a Chinese error when not a TTY
  - guided_setup(setup): the interactive guide engine ("press Enter to continue"; only risky branches get a multiple-choice menu)

Design principle: beginners want a "default next step" rather than "options". Things
with a single answer, like a missing dependency or missing login, default to Enter;
branches with side effects, like version conflicts, get a menu. Interaction is only
triggered in a real terminal (both sys.stdin and sys.stdout are TTYs); CI / pytest /
Jupyter always fall back to an exception.

Declarative setup structure (using Quantum Inspire as an example):
    {
        "name": "Quantum Inspire",              # display name
        "sdk": {
            "package": "qiskit_quantuminspire", # import name, used for find_spec detection
            "pip": "qiskit-quantuminspire",     # PyPI name, used for messages / version queries
            "install": "quonic[quantum-inspire]",  # recommended install command (may include extras)
        },
        "auth": {
            "kind": "oauth_cli",
            "command": ["qi", "login"],
            "token_file": "~/.quantuminspire/config.json",
        },
        "conflicts": [{"package": "qiskit", "constraint": "<2.4.0"}],
        "devices": ["Tuna-9", "Tuna-17", "QX emulator"],
        "billing": True,                         # real-hardware billing → confirm before submission
    }
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .._i18n import tr

# ---------------------------------------------------------------------------
# Diagnostics (pure functions, unit-testable)
# ---------------------------------------------------------------------------

@dataclass
class Diagnosis:
    sdk_installed: bool
    conflicts: list[dict[str, str]] = field(default_factory=list)
    auth_ready: bool = False

    @property
    def ready(self) -> bool:
        return self.sdk_installed and not self.conflicts and self.auth_ready


def _module_available(package: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(package) is not None


def _installed_version(package: str) -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_tuple(v: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(v))
    return tuple(int(p) for p in parts)


def _compare(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Compare two version tuples and return -1 / 0 / 1. Pad lengths, then compare digit by digit."""
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


def satisfies(current: str, constraint: str) -> bool:
    """Return whether version current satisfies a constraint such as '<2.4.0' / '>=1.0' / '==2.0'."""
    m = re.match(r"(<=|>=|==|<|>)\s*([\d.]+)", str(constraint))
    if not m:
        raise ValueError(tr("err.parse_constraint", constraint=constraint))
    op, ver = m.group(1), m.group(2)
    c = _compare(_version_tuple(current), _version_tuple(ver))
    return {"<": c < 0, "<=": c <= 0, ">": c > 0, ">=": c >= 0, "==": c == 0}[op]


def _auth_ready(auth: dict[str, Any]) -> bool:
    import os

    token_file = auth.get("token_file")
    if not token_file:
        return False
    return os.path.isfile(os.path.expanduser(token_file))


def _conflicts(setup: dict[str, Any]) -> list[dict[str, str]]:
    """Return the list of conflicts that violate constraints: [{"package", "installed", "constraint"}]."""
    conflicts: list[dict[str, str]] = []
    for c in setup.get("conflicts", []):
        ver = _installed_version(c["package"])
        if ver is not None and not satisfies(ver, c["constraint"]):
            conflicts.append(
                {"package": c["package"], "installed": ver, "constraint": c["constraint"]}
            )
    return conflicts


def diagnose(setup: dict[str, Any]) -> Diagnosis:
    """Diagnose whether the backend described by setup is ready, returning a Diagnosis."""
    sdk = setup["sdk"]
    return Diagnosis(
        sdk_installed=_module_available(sdk["package"]),
        conflicts=_conflicts(setup),
        auth_ready=_auth_ready(setup.get("auth", {})),
    )


# ---------------------------------------------------------------------------
# Interactive helpers (input_ / run are injectable for testing)
# ---------------------------------------------------------------------------

def _is_interactive() -> bool:
    import sys

    stdin_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    stdout_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    return stdin_tty and stdout_tty


def _confirm(prompt: str, input_: Callable[[str], str] = input) -> bool:
    """Read a line: Enter (empty line) means "yes", n/no means "no"; EOF means no."""
    try:
        ans = input_(prompt + tr("setup.confirm_hint"))
    except EOFError:
        return False
    return ans.strip().lower() not in ("n", "no")


def _menu(
    prompt: str, input_: Callable[[str], str] = input, default: str = "1"
) -> str:
    try:
        ans = input_(prompt + tr("setup.menu_suffix"))
    except EOFError:
        return default
    return ans.strip() or default


def _run_pip(run: Callable[..., Any], target: str) -> Any:
    import sys

    return run([sys.executable, "-m", "pip", "install", target])


# ---------------------------------------------------------------------------
# Guide engine
# ---------------------------------------------------------------------------

def guided_setup(
    setup: dict[str, Any],
    input_: Callable[[str], str] = input,
    run: Callable[..., Any] | None = None,
) -> bool:
    """Interactively guide configuration. Returns whether it is ready. Should only be called in a TTY environment."""
    if run is None:
        import subprocess

        run = subprocess.run

    name = setup.get("name", tr("setup.default_name"))
    print(tr("setup.configuring", name=name))

    # [1/3] Dependency
    sdk = setup["sdk"]
    if not _module_available(sdk["package"]):
        print(tr("setup.missing_dep", pkg=sdk.get("pip", sdk["package"])))
        print(tr("setup.will_run", install=sdk["install"]))
        if _confirm(tr("setup.press_enter_install"), input_):
            _run_pip(run, sdk["install"])

    # [2/3] Version conflicts (only checked after the dependency is ready, to avoid false reports of missing packages)
    if _module_available(sdk["package"]):
        for c in _conflicts(setup):
            print(
                tr(
                    "setup.conflict_detected",
                    pkg=c["package"],
                    installed=c["installed"],
                    constraint=c["constraint"],
                )
            )
            _handle_conflict(c, run, input_)

    # [3/3] Authentication
    auth = setup.get("auth")
    if auth and not _auth_ready(auth):
        cmd = " ".join(auth.get("command", [])) or tr("setup.login_fallback")
        print(tr("setup.need_login", cmd=cmd))
        if _confirm(tr("setup.press_enter_login"), input_):
            run(auth["command"])
            if _auth_ready(auth):
                print(tr("setup.logged_in"))
            else:
                print(tr("setup.login_incomplete", cmd=cmd))

    d = diagnose(setup)
    if d.ready:
        print(tr("setup.ready", name=name))
    else:
        print(tr("setup.not_ready"))
    return d.ready


def _handle_conflict(
    c: dict[str, str], run: Callable[..., Any], input_: Callable[[str], str]
) -> None:
    print(tr("setup.how_to_handle"))
    print(tr("setup.opt_venv"))
    print(tr("setup.opt_downgrade", pkg=c["package"], constraint=c["constraint"]))
    print(tr("setup.opt_skip"))
    ans = _menu(tr("setup.prompt_input"), input_, default="1")
    if ans == "1":
        _print_venv_guide()
        return
    if ans == "3":
        return
    print(tr("setup.will_run_pip", pkg=c["package"], constraint=c["constraint"]))
    _run_pip(run, f"{c['package']}{c['constraint']}")


def _print_venv_guide() -> None:
    import platform

    print(tr("setup.venv_create"))
    print("        python -m venv .venv-qi")
    if platform.system() == "Windows":
        print(r"        .venv-qi\Scripts\activate")
    else:
        print("        source .venv-qi/bin/activate")
    print("        pip install 'quonic[quantum-inspire]'")
    print("        qi login")
    print(tr("setup.venv_rerun"))


# ---------------------------------------------------------------------------
# Preflight check called by backends' run()
# ---------------------------------------------------------------------------

def ensure_ready(
    setup: dict[str, Any],
    input_: Callable[[str], str] = input,
    run: Callable[..., Any] | None = None,
) -> None:
    """Preflight check for a backend's run(): pass when ready; otherwise guide under a TTY, and raise a Chinese error when not a TTY.

    Non-interactive environments (pytest / CI / Jupyter) never prompt; they raise
    immediately, keeping tests and automation dependable.
    """
    d = diagnose(setup)
    if d.ready:
        return
    if _is_interactive():
        guided_setup(setup, input_=input_, run=run)
        d = diagnose(setup)
    if d.ready:
        return
    _raise_not_ready(d, setup)


def _raise_not_ready(d: Diagnosis, setup: dict[str, Any]) -> None:
    name = setup.get("name", tr("setup.default_name"))
    if not d.sdk_installed:
        sdk = setup["sdk"]
        raise ImportError(
            tr(
                "err.need_install",
                name=name,
                pkg=sdk.get("pip", sdk["package"]),
                install=sdk["install"],
            )
        )
    for c in d.conflicts:
        raise RuntimeError(
            tr(
                "err.conflict",
                name=name,
                pkg=c["package"],
                installed=c["installed"],
                constraint=c["constraint"],
            )
        )
    auth = setup.get("auth", {})
    cmd = " ".join(auth.get("command", [])) or tr("setup.login_fallback")
    raise RuntimeError(tr("err.need_login", name=name, cmd=cmd))


__all__ = [
    "Diagnosis",
    "diagnose",
    "ensure_ready",
    "guided_setup",
    "satisfies",
]
