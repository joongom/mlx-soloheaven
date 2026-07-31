"""Model architectures that upstream mlx-lm does not (yet) ship.

mlx-lm dispatches on ``config.json``'s ``model_type`` via
``importlib.import_module(f"mlx_lm.models.{model_type}")`` (see
``mlx_lm/utils.py::_get_classes``). Since ``import_module`` consults
``sys.modules`` first, we can teach mlx-lm about an architecture by
pre-registering our module under the name it will look up — no monkeypatching
of mlx-lm internals, and no editing files inside ``.venv`` (which any
``pip install -U mlx-lm`` would wipe out).

Upstream always wins: if a future mlx-lm release ships the architecture for
real, ``register_extra_architectures()`` imports theirs and leaves ours unused.

Call ``register_extra_architectures()`` before any ``mlx_lm.utils.load`` /
``mlx_lm.convert.convert``. It is idempotent and cheap.
"""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

# model_type (as it appears in config.json) -> module implementing it here.
_EXTRA_ARCHITECTURES: dict[str, str] = {
    # EXAONE-4.5 (LGAI-EXAONE). Text tower only — see exaone4_5.py.
    "exaone4_5": "mlx_soloheaven.models.exaone4_5",
}

_registered: dict[str, str] = {}


def register_extra_architectures() -> dict[str, str]:
    """Register our extra architectures into the ``mlx_lm.models`` namespace.

    Returns a mapping of ``model_type -> module name`` for the entries this
    process actually serves (i.e. excluding any that upstream mlx-lm provides).
    Idempotent: repeated calls are no-ops.
    """
    for model_type, module_path in _EXTRA_ARCHITECTURES.items():
        target = f"mlx_lm.models.{model_type}"
        if target in sys.modules:
            continue

        # Prefer upstream if it exists in the installed mlx-lm.
        try:
            importlib.import_module(target)
            logger.debug(
                "[models] mlx-lm already provides %r — not registering ours",
                model_type,
            )
            continue
        except ImportError:
            pass

        module = importlib.import_module(module_path)
        sys.modules[target] = module
        # Also bind as an attribute so `from mlx_lm.models import <t>` works.
        try:
            setattr(importlib.import_module("mlx_lm.models"), model_type, module)
        except Exception:  # noqa: BLE001 — attribute binding is best-effort
            pass

        _registered[model_type] = module_path
        logger.info(
            "[models] registered %r -> %s (not in installed mlx-lm)",
            model_type,
            module_path,
        )

    return dict(_registered)


__all__ = ["register_extra_architectures"]
