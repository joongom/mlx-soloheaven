#!/usr/bin/env python
"""Convert an LGAI-EXAONE EXAONE-4.5 checkpoint to a quantized MLX model.

    ./convert_exaone4_5.py                      # 8-bit (default), from the HF repo
    ./convert_exaone4_5.py --q-bits 4
    ./convert_exaone4_5.py --hf-path /local/EXAONE-4.5-33B --mlx-path /out/dir

Why this wrapper instead of plain ``mlx_lm.convert``:

  1. ``model_type: exaone4_5`` is not in mlx-lm, so ``mlx_lm.convert`` fails at
     load with "Model type exaone4_5 not supported". We register
     ``mlx_soloheaven.models.exaone4_5`` first (see that module for the
     architecture analysis).

  2. ``mlx_lm.convert`` writes back the *source* config verbatim, so the output
     would keep the nested ``exaone4_5`` shape and stay dependent on our
     registration forever. We rewrite it into a flat ``exaone4`` config
     afterwards: the text tower IS EXAONE-4.0's architecture, the saved tensor
     names already match ``mlx_lm/models/exaone4.py``'s module tree, so the
     result is an ordinary mlx-lm model that loads with stock mlx-lm, LM Studio,
     ``mlx_lm.generate`` — no shim, and immune to mlx-lm upgrades.
     Pass ``--no-flatten`` to keep the nested config instead.

TEXT ONLY. EXAONE-4.5 is multimodal; the vision tower (``model.visual.*``) and
the multi-token-prediction head (``mtp.*``) are dropped during conversion.
Serving vision would require mlx-vlm support, which does not exist for EXAONE.

Peak memory: the bf16 source is ~69 GB and quantization materializes on top of
it. Close other large processes before running on a 128 GB machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mlx_soloheaven.models import register_extra_architectures  # noqa: E402
from mlx_soloheaven.models.exaone4_5 import flatten_config  # noqa: E402

DEFAULT_HF_PATH = "LGAI-EXAONE/EXAONE-4.5-33B"

# Config keys that only make sense for the multimodal wrapper / MTP head, both
# of which we drop. Leaving them behind would describe weights that aren't
# there — transformers would try to build towers the checkpoint no longer has.
_DROP_KEYS = (
    "text_config",
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "_num_mtp_layers",
    "num_nextn_predict_layers",
    "mtp_loss_scaling_factor",
    "mtp_share_layers",
)


def flatten_saved_config(mlx_path: Path, hf_path: str) -> None:
    """Rewrite the converted config.json as a flat, stock-loadable exaone4."""
    config_path = mlx_path / "config.json"
    config = json.loads(config_path.read_text())

    flat = flatten_config(config)
    # Quantization metadata is added by mlx_lm.convert and keys into the MLX
    # module tree, which our sanitize() already renamed to exaone4's layout.
    for key in ("quantization", "quantization_config"):
        if key in config:
            flat[key] = config[key]
    for key in _DROP_KEYS:
        flat.pop(key, None)

    flat["model_type"] = "exaone4"
    flat["architectures"] = ["Exaone4ForCausalLM"]
    flat["_converted_from"] = {
        "source": hf_path,
        "source_model_type": config.get("model_type", "exaone4_5"),
        "converter": "mlx-soloheaven/convert_exaone4_5.py",
        "note": (
            "Text tower only. The EXAONE-4.5 vision tower (model.visual.*) and "
            "MTP head (mtp.*) were dropped; the text tower is architecturally "
            "identical to EXAONE-4.0, hence model_type=exaone4."
        ),
    }

    config_path.write_text(json.dumps(flat, indent=2, ensure_ascii=False) + "\n")
    print(f"[INFO] Flattened config.json -> model_type=exaone4 ({config_path})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert EXAONE-4.5 to quantized MLX (text tower only)."
    )
    parser.add_argument("--hf-path", default=DEFAULT_HF_PATH)
    parser.add_argument(
        "--mlx-path",
        default=None,
        help="Output dir (default: ~/.lmstudio/models/mlx-community/"
        "<repo-name>-<bits>bit)",
    )
    parser.add_argument("--q-bits", type=int, default=8)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument(
        "--no-flatten",
        action="store_true",
        help="Keep the nested exaone4_5 config (needs this repo to load it).",
    )
    args = parser.parse_args()

    mlx_path = Path(
        args.mlx_path
        or (
            Path.home()
            / ".lmstudio/models/mlx-community"
            / f"{args.hf_path.rstrip('/').split('/')[-1]}-{args.q_bits}bit"
        )
    ).expanduser()

    registered = register_extra_architectures()
    print(f"[INFO] Extra architectures registered: {registered or '(none needed)'}")

    from mlx_lm.convert import convert

    convert(
        hf_path=args.hf_path,
        mlx_path=str(mlx_path),
        quantize=True,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )

    if not args.no_flatten:
        flatten_saved_config(mlx_path, args.hf_path)

    print(f"[DONE] {mlx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
