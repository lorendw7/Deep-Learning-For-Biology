"""Check the pinned environment with small examples from all five chapters."""

import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import pkgutil
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ASSETS_DIR", str(ROOT / "data" / "assets"))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".uv-cache" / "matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def check_versions():
    from packaging.requirements import Requirement

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Run this check with the project's Python 3.11 environment.")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for text in config["project"]["dependencies"]:
        requirement = Requirement(text)
        version = metadata.version(requirement.name)
        if not requirement.specifier.contains(version):
            raise RuntimeError(f"Wrong version: {requirement.name} {version}")
    source = json.loads(metadata.distribution("dlfb").read_text("direct_url.json"))
    assert source["vcs_info"]["commit_id"] == config["tool"]["uv"]["sources"]["dlfb"]["rev"]
    print(f"Python: {sys.version.split()[0]} at {sys.executable}", flush=True)


def check_imports():
    from book_runtime import configure

    configure(ROOT)
    for name in (
        "Bio", "adjustText", "datasets", "dm_pix", "flax", "jax", "jraph",
        "matplotlib.pyplot", "ml_collections", "numpy", "obonet", "ogb",
        "optax", "orbax.checkpoint", "pandas", "parfive", "py3Dmol", "scipy",
        "seaborn", "sklearn", "tensorflow", "tensorflow_datasets", "torch",
        "torchvision", "transformers", "umap",
    ):
        importlib.import_module(name)
    import dlfb

    # The official cancer module loads three pretrained ResNets on import.
    # Their first download is saved under data/huggingface.
    modules = list(pkgutil.walk_packages(dlfb.__path__, dlfb.__name__ + "."))
    for module in modules:
        importlib.import_module(module.name)
    print(f"Imported all {len(modules)} dlfb modules.", flush=True)


def check_proteins():
    import jax
    import numpy as np
    import optax
    import pandas as pd
    from dlfb.proteins.dataset import convert_to_tfds
    from dlfb.proteins.model import Model
    from dlfb.proteins.train import train

    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(8, 8)).astype("float32"),
                         columns=[f"ME:{i}" for i in range(8)])
    frame[["GO:0", "GO:1"]] = np.tile([[1.0, 0.0], [0.0, 1.0]], (4, 1)).astype("float32")
    splits = {"train": convert_to_tfds(frame, is_training=True),
              "valid": convert_to_tfds(frame)}
    state = Model(num_targets=2, dim=8).create_train_state(
        jax.random.PRNGKey(0), np.zeros((4, 8), dtype="float32"), optax.adam(0.01)
    )
    initial = state
    state, metrics = train(state, splits, batch_size=4, num_steps=3, eval_every=3)
    assert int(state.step) == 3
    assert all(np.isfinite(row["loss"]) for row in metrics["train"])
    assert any(not np.array_equal(a, b) for a, b in
               zip(jax.tree.leaves(initial.params), jax.tree.leaves(state.params)))
    return state


def check_dna():
    import jax
    import numpy as np
    import optax
    from dlfb.dna.dataset import convert_to_tfds
    from dlfb.dna.model import ConvModelV2
    from dlfb.dna.train import train_step
    from dlfb.dna.utils import dna_to_one_hot, one_hot_to_dna

    sequence = "ACGT" * 4
    encoded = dna_to_one_hot(sequence)
    assert one_hot_to_dna(encoded) == sequence
    data = {"sequences": np.stack([encoded, encoded[::-1]]).astype("float32"),
            "labels": np.array([[0.0], [1.0]], dtype="float32")}
    batch = next(convert_to_tfds(data, batch_size=2).as_numpy_iterator())
    model = ConvModelV2(conv_filters=4, kernel_size=(3,), dense_units=8, dropout_rate=0.0)
    state = model.create_train_state(jax.random.PRNGKey(1), batch["sequences"], optax.adam(0.001))
    state, metrics = train_step(state, batch, jax.random.PRNGKey(2))
    assert int(state.step) == 1 and np.isfinite(float(metrics["loss"]))


def check_graphs():
    import jax
    import jax.numpy as jnp
    import jraph
    import numpy as np
    import optax
    from dlfb.graphs.model import DdiModel
    from dlfb.graphs.train import train_step

    graph = jraph.GraphsTuple(
        nodes={"gid": jnp.arange(4)}, edges=None, globals=None,
        senders=jnp.array([0, 1, 2, 3]), receivers=jnp.array([1, 2, 3, 0]),
        n_node=jnp.array([4]), n_edge=jnp.array([4]),
    )
    pairs = {"pos": jnp.array([[0, 1], [2, 3]]),
             "neg": jnp.tile(jnp.array([[0, 2], [1, 3]]), (12, 1))}
    model = DdiModel(n_nodes=4, embedding_dim=8, dropout_rate=0.0,
                     last_layer_self=True, degree_norm=False)
    state = model.create_train_state(jax.random.PRNGKey(3),
                                     {"graph": graph, "pairs": pairs}, optax.adam(0.001))
    state, metrics = train_step(state, graph, pairs, jax.random.PRNGKey(4))
    assert int(state.step) == 1
    assert all(np.isfinite(float(value)) for value in metrics.values())


def check_cancer():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from dlfb.cancer.model.cnn import SimpleCnn
    from dlfb.cancer.train import train_step

    images = jax.random.uniform(jax.random.PRNGKey(5), (2, 16, 16, 3))
    model = SimpleCnn(num_classes=2)
    state = model.create_train_state(jax.random.PRNGKey(6), images, optax.adam(0.001))
    state, metrics = train_step(state, {"images": images, "labels": jnp.array([0, 1])},
                                jax.random.PRNGKey(7), 2)
    assert int(state.step) == 1 and np.isfinite(float(metrics["loss"]))


def check_localization():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from dlfb.localization.model import LocalizationModel
    from dlfb.localization.train import train_step
    from umap import UMAP

    images = jax.random.uniform(jax.random.PRNGKey(8), (2, 16, 16, 1))
    model = LocalizationModel(embedding_dim=32, num_embeddings=4, commitment_cost=0.25,
                              num_classes=2, dropout_rate=0.0, classification_head_layers=1)
    state = model.create_train_state(jax.random.PRNGKey(9), images, optax.adam(0.001))
    output = state.apply_fn({"params": state.params}, images, is_training=False)
    assert output[0].shape == images.shape
    state, metrics = train_step(state, {"images": images, "labels": jnp.array([0, 1])},
                                jax.random.PRNGKey(10), 1.0)
    assert int(state.step) == 1
    assert all(np.isfinite(float(value)) for value in metrics.values())
    embedding = UMAP(n_neighbors=5, random_state=42, n_jobs=1).fit_transform(
        np.random.default_rng(42).normal(size=(30, 4)).astype("float32")
    )
    assert embedding.shape == (30, 2) and np.isfinite(embedding).all()


def check_esm():
    import torch
    from transformers import EsmConfig, EsmModel
    from torchvision.ops import nms

    model = EsmModel(EsmConfig(vocab_size=33, hidden_size=16, num_hidden_layers=1,
                              num_attention_heads=2, intermediate_size=32,
                              pad_token_id=1, mask_token_id=32))
    output = model(input_ids=torch.tensor([[0, 5, 6, 2]]))
    assert tuple(output.last_hidden_state.shape) == (1, 4, 16)
    output.last_hidden_state.square().mean().backward()
    assert model.embeddings.word_embeddings.weight.grad is not None
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
    assert nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]


def check_checkpoint(state):
    import jax
    import numpy as np
    from dlfb.utils.restore import restore, store

    output = ROOT / "outputs" / "verification"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as directory:
        path = str(Path(directory) / "checkpoint")
        store(path, state, {"check": "windows"})
        restored, metrics = restore(path, state)
        assert metrics == {"check": "windows"}
        for expected, actual in zip(jax.tree.leaves(state.params), jax.tree.leaves(restored.params)):
            np.testing.assert_allclose(expected, actual)


def check_jupyter():
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        f"import sys, book_runtime\nbook_runtime.configure({str(ROOT)!r})\n"
        f"import jax\nassert sys.executable == {sys.executable!r}\n"
        "assert jax.numpy.ones((2, 2)).sum().item() == 4\nprint('Kernel check passed')"
    )])
    NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=str(ROOT))


def main():
    checks = [check_versions, check_imports, check_proteins, check_dna, check_graphs,
              check_cancer, check_localization, check_esm]
    state = None
    for check in checks:
        print(f"Running {check.__name__}...", flush=True)
        result = check()
        if check is check_proteins:
            state = result
        print(f"PASS: {check.__name__}", flush=True)
    print("Running check_checkpoint...", flush=True)
    check_checkpoint(state)
    print("PASS: check_checkpoint", flush=True)
    print("Running check_jupyter...", flush=True)
    check_jupyter()
    print("PASS: check_jupyter", flush=True)
    print("All environment checks passed. Full datasets and long training runs were not tested.", flush=True)


if __name__ == "__main__":
    main()
