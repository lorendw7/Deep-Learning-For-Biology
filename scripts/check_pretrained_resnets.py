"""Compare the book's repaired Flax ResNets against their PyTorch checkpoints."""

import argparse
import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_REPO = "huggingface/documentation-images"
IMAGE_REVISION = "2dd75e791509da38222ecbd3bfaa2b44ac658ef4"
PROCESSOR_REVISION = "34c2154c194f829b11125337b98c8f5f9965ff19"


class WarningRecorder(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def check_all_tensors(flax_model, torch_model):
    """Independently compare every Flax leaf with the original PyTorch tensor."""
    import numpy as np
    from flax.traverse_util import flatten_dict

    original = torch_model.state_dict()
    checked_keys = set()
    for key, value in flatten_dict(flax_model.params).items():
        collection, *path = key
        path = ["0" if part == "layer_0" else "1" if part == "layer_1" else part for part in path]
        tensor = np.asarray(value)
        if path[-1] == "kernel":
            path[-1] = "weight"
            tensor = tensor.transpose(3, 2, 0, 1) if tensor.ndim == 4 else tensor.T
        elif path[-1] == "scale":
            path[-1] = "weight"
        elif collection == "batch_stats":
            path[-1] = {"mean": "running_mean", "var": "running_var"}[path[-1]]
        source_key = "resnet." + ".".join(path)
        np.testing.assert_array_equal(tensor, original[source_key].cpu().numpy(), err_msg=source_key)
        checked_keys.add(source_key)
    backbone_keys = {key for key in original if key.startswith("resnet.") and not key.endswith("num_batches_tracked")}
    assert checked_keys == backbone_keys
    return len(checked_keys)


def check_rejected_states():
    import numpy as np
    from resnet_compat import _validate_state

    key = ("params", "weight")
    expected = {key: np.ones((2, 2), dtype="float32")}
    bad_states = [{}, {**expected, ("params", "extra"): np.ones(1)},
                  {key: np.ones((1, 2))}, {key: np.full((2, 2), np.nan)}]
    for state in bad_states:
        try:
            _validate_state(expected, state)
        except ValueError:
            continue
        raise AssertionError("A broken checkpoint state was accepted.")


def check_book_model(depth, pretrained, pixels):
    import jax
    import numpy as np
    import optax
    from dlfb.cancer.model.resnet import FinetunedResNet
    from flax.traverse_util import flatten_dict

    images = pixels[:1].transpose(0, 2, 3, 1)
    model = FinetunedResNet(num_classes=2, layers=depth)
    state = model.create_train_state(jax.random.PRNGKey(42), images, optax.sgd(0.01))
    for collection in ("params", "batch_stats"):
        actual = state.params["backbone"] if collection == "params" else state.batch_stats["backbone"]
        expected = flatten_dict(pretrained.params[collection])
        loaded = flatten_dict(actual)
        assert expected.keys() == loaded.keys()
        for key in expected:
            np.testing.assert_array_equal(expected[key], loaded[key])
    output = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, images, is_training=False)
    assert output.shape == (1, 2) and np.isfinite(output).all()
    return list(output.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use only files already in the local cache.")
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from book_runtime import configure
    configure(ROOT)

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from resnet_compat import CHECKPOINTS
    from transformers import AutoImageProcessor, ResNetForImageClassification

    check_rejected_states()
    recorder = WarningRecorder()
    logger = logging.getLogger("transformers")
    logger.addHandler(recorder)
    try:
        # Import through the actual book registry, rather than bypassing it.
        from dlfb.cancer.model import PRETRAINED_RESNETS

        image_path = hf_hub_download(IMAGE_REPO, "coco_sample.png", repo_type="dataset",
                                    revision=IMAGE_REVISION, local_files_only=args.offline)
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", revision=PROCESSOR_REVISION,
                                                       use_fast=False, local_files_only=args.offline)
        with Image.open(image_path) as image:
            photo_pixels = processor(images=image.convert("RGB"), return_tensors="np")["pixel_values"]
        noise = np.random.default_rng(42).normal(size=photo_pixels.shape).astype("float32")
        pixels = np.concatenate([photo_pixels, noise], axis=0)
        results = []
        for depth in CHECKPOINTS:
            repo_id, revision = CHECKPOINTS[depth]
            print(f"Checking {repo_id} at {revision}...", flush=True)
            model = PRETRAINED_RESNETS[depth]
            info = model.book_loading_info
            assert not any(info[name] for name in ("missing_keys", "unexpected_keys", "mismatched_keys"))
            reference, loading = ResNetForImageClassification.from_pretrained(
                repo_id, revision=revision, use_safetensors=True,
                local_files_only=args.offline, output_loading_info=True,
            )
            assert not any(loading[name] for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
            reference.eval()
            tensors = check_all_tensors(model, reference)
            with torch.inference_mode():
                torch_output = reference(pixel_values=torch.from_numpy(pixels), output_hidden_states=True)
            flax_output = model(pixel_values=pixels, train=False)
            features = np.asarray(flax_output.last_hidden_state)
            torch_features = torch_output.hidden_states[-1].cpu().numpy()
            np.testing.assert_allclose(features, torch_features, rtol=1e-4, atol=1e-4)
            pooled = np.asarray(flax_output.pooler_output).reshape(len(pixels), -1)
            logits = pooled @ reference.classifier[1].weight.detach().cpu().numpy().T
            logits += reference.classifier[1].bias.detach().cpu().numpy()
            torch_logits = torch_output.logits.cpu().numpy()
            assert np.isfinite(logits).all()
            np.testing.assert_allclose(logits, torch_logits, rtol=1e-4, atol=1e-4)
            top5 = np.argsort(logits, axis=-1)[:, -5:][:, ::-1]
            np.testing.assert_array_equal(top5, np.argsort(torch_logits, axis=-1)[:, -5:][:, ::-1])
            book_shape = check_book_model(depth, model, pixels)
            result = {
                **info, "exactly_matched_tensors": tensors,
                "feature_max_absolute_error": float(np.abs(features - torch_features).max()),
                "logit_max_absolute_error": float(np.abs(logits - torch_logits).max()),
                "photo_top5_labels": [reference.config.id2label[int(index)] for index in top5[0]],
                "top5_matches_pytorch": True, "book_inference_shape": book_shape,
            }
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
        if recorder.messages:
            raise AssertionError(f"Transformers loading warnings remain: {recorder.messages}")
    finally:
        logger.removeHandler(recorder)

    output = ROOT / "outputs" / "verification" / "pretrained-resnets.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"models": results, "transformers_warnings": recorder.messages,
                                  "reference_image_revision": IMAGE_REVISION}, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: ResNet-18/34 weights, reference inference, and book integration. Report: {output}", flush=True)


if __name__ == "__main__":
    main()
