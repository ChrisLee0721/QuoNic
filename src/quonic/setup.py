"""python -m quonic.setup — one-click onboarding entry for real-hardware backends.

By default it onboards Quantum Inspire; use --backend to onboard other hardware backends.
"""

from __future__ import annotations

import argparse

from ._i18n import tr
from .backends.originq import OriginQBackend
from .backends.qi import QuantumInspireBackend
from .backends.setup_guide import diagnose, guided_setup

_BACKEND_SETUPS = {
    "qi": QuantumInspireBackend.setup,
    "originq": OriginQBackend.setup,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="QuoNic hardware backend onboarding")
    parser.add_argument("--backend", default="qi", choices=list(_BACKEND_SETUPS),
                        help="Backend to onboard (default: qi)")
    args = parser.parse_args()

    setup = _BACKEND_SETUPS[args.backend]

    if diagnose(setup).ready:
        print(tr("setup.ready", name=setup["name"]))
        return 0

    ok = guided_setup(setup)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
