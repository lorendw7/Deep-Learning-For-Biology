"""Strictly convert the book's pinned ResNet-18/34 checkpoints to Flax."""

from functools import wraps
from importlib.metadata import version
import sys


CHECKPOINTS = {
    18: ("microsoft/resnet-18", "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"),
    34: ("microsoft/resnet-34", "4d180fbdc81b4d5d4eaa86ab89bdd669200dd477"),
}


def _validate_state(expected, loaded):
    """Reject incomplete conversions instead of filling gaps with random weights."""
    import numpy as np

    missing = expected.keys() - loaded.keys()
    unexpected = loaded.keys() - expected.keys()
    if missing or unexpected:
        raise ValueError(f"Incomplete ResNet conversion: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, tensor in loaded.items():
        if tensor.shape != expected[key].shape:
            raise ValueError(f"Wrong ResNet shape for {key}: {tensor.shape} != {expected[key].shape}")
        if not np.isfinite(np.asarray(tensor)).all():
            raise ValueError(f"Non-finite ResNet checkpoint tensor: {key}")


def load_pretrained_resnet(
    depth, *, cache_dir=None, local_files_only=False, revision=None,
    force_download=False, token=None, dtype=None, seed=0, input_shape=(1, 224, 224, 3),
):
    """Load every backbone weight and BatchNorm state from a pinned checkpoint."""
    import jax.numpy as jnp
    from flax.traverse_util import flatten_dict
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import FlaxResNetModel, ResNetConfig
    from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax

    if version("transformers") != "4.49.0":
        raise RuntimeError("This ResNet adapter is verified with Transformers 4.49.0. Run uv sync --locked.")
    repo_id, pinned_revision = CHECKPOINTS[depth]
    if revision is not None and revision != pinned_revision:
        raise ValueError(f"Use the verified checkpoint revision {pinned_revision} for {repo_id}.")
    options = dict(repo_id=repo_id, revision=pinned_revision, cache_dir=cache_dir,
                   local_files_only=local_files_only, force_download=force_download, token=token)
    config_path = hf_hub_download(filename="config.json", **options)
    weights_path = hf_hub_download(filename="model.safetensors", **options)
    config = ResNetConfig.from_json_file(config_path)
    model = FlaxResNetModel(config, dtype=dtype or jnp.float32, seed=seed, input_shape=input_shape)

    state = {}
    classifier_keys = {"classifier.1.weight", "classifier.1.bias"}
    ignored_classifier = []
    ignored_counters = []
    with safe_open(weights_path, framework="np") as checkpoint:
        if (checkpoint.metadata() or {}).get("format") != "pt":
            raise ValueError("Expected a PyTorch safetensors checkpoint.")
        for key in checkpoint.keys():
            if key in classifier_keys:
                # dlfb replaces the ImageNet classifier with a skin-lesion head.
                ignored_classifier.append(key)
                continue
            if not key.startswith("resnet."):
                raise ValueError(f"Unexpected checkpoint tensor outside the ResNet backbone: {key}")
            if key.endswith(".num_batches_tracked"):
                ignored_counters.append(key)
            # Flax's unnamed basic-block layers become layer_0 and layer_1.
            # Matching these names also lets the standard converter recognize
            # BatchNorm weight -> scale, rather than treating it as a dense kernel.
            target = key.replace(".layer.0.", ".layer.layer_0.").replace(".layer.1.", ".layer.layer_1.")
            if target in state:
                raise ValueError(f"Duplicate converted checkpoint key: {target}")
            state[target] = checkpoint.get_tensor(key)

    if set(ignored_classifier) != classifier_keys:
        raise ValueError("The checkpoint does not contain the expected ImageNet classifier.")
    converted = convert_pytorch_state_dict_to_flax(state, model)
    expected, loaded = flatten_dict(model.params), flatten_dict(converted)
    _validate_state(expected, loaded)
    model.params = converted
    model.book_loading_info = {
        "repo_id": repo_id, "revision": pinned_revision, "loaded_tensors": len(loaded),
        "missing_keys": [], "unexpected_keys": [], "mismatched_keys": [],
        "excluded_imagenet_head": sorted(ignored_classifier),
        "excluded_pytorch_counters": len(ignored_counters),
    }
    return model


def install_book_loader():
    """Route the book's two default model IDs through the checked conversion."""
    from transformers import FlaxResNetModel

    original = FlaxResNetModel.from_pretrained.__func__
    if getattr(original, "_book_resnet_loader", False):
        return
    existing = sys.modules.get("dlfb.cancer.model")
    if existing is not None:
        raise RuntimeError("Restart Python and call book_runtime.configure() before importing dlfb.cancer.model.")
    depths = {repo_id: depth for depth, (repo_id, _) in CHECKPOINTS.items()}

    @wraps(original)
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if cls is FlaxResNetModel and str(pretrained_model_name_or_path) in depths:
            if args:
                raise TypeError("Use keyword arguments with the book's ResNet loader.")
            return load_pretrained_resnet(depths[str(pretrained_model_name_or_path)], **kwargs)
        return original(cls, pretrained_model_name_or_path, *args, **kwargs)

    from_pretrained._book_resnet_loader = True
    FlaxResNetModel.from_pretrained = classmethod(from_pretrained)
