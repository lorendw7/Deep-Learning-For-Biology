"""Set local paths, load libraries in order, and enable the book's model fixes."""

import os
from pathlib import Path
import sys


def configure(project_root: str | Path | None = None) -> Path:
    """Call this before importing JAX, Flax, Transformers, or dlfb."""
    root = Path(project_root or Path(__file__).parent).resolve()
    assets = root / "data" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ASSETS_DIR", str(assets))
    os.environ.setdefault("HF_HOME", str(root / "data" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if "jax" in sys.modules and "tensorflow" not in sys.modules:
        raise RuntimeError("Restart Python and call book_runtime.configure() before importing JAX.")
    # TensorFlow 2.19 fails to load its native DLL after JAX 0.5.1 on this Windows host.
    import tensorflow  # noqa: F401

    from resnet_compat import install_book_loader

    install_book_loader()
    return root
