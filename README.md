# Deep Learning for Biology

This is my personal study repository for *Deep Learning for Biology: Harness AI
to Solve Real-World Biology Problems* by Charles Ravarani and Natasha Latysheva
(O'Reilly).

I use this repository to work through the book, write code, and keep short notes.
All documentation and study notes use plain, simple English.

## Environment

This project uses **Python 3.11 on 64-bit Windows**, with CPU versions of JAX,
TensorFlow, and PyTorch. The dependencies cover the five official project chapters
(chapters 2 to 6). They follow these pinned upstream sources:

- [Supporting library and requirements](https://github.com/deep-learning-for-biology/dlfb/tree/34809479cab19295902dcd70159dbb07f6ad403f), commit `3480947`.
- [Official notebooks](https://github.com/deep-learning-for-biology/notebooks/tree/c705571f6041a97c45fefbb2490c4a83e81a5672), commit `c705571`.

The main versions match the official requirements:

| Package | Version |
| --- | --- |
| NumPy / pandas | 2.1.2 / 2.2.2 |
| scikit-learn / Matplotlib | 1.4.2 / 3.10.1 |
| JAX / jaxlib | 0.5.1 / 0.5.1 |
| Flax / Optax | 0.10.4 / 0.2.4 |
| TensorFlow / TensorFlow Datasets | 2.19.0 / 4.9.8 |
| PyTorch / torchvision | 2.6.0+cpu / 0.21.0+cpu |
| Transformers / Hugging Face Hub | 4.49.0 / 0.36.0 |
| Biopython | 1.85 |

`pyproject.toml` lists the direct dependencies. `uv.lock` records the full resolved
environment, including the exact supporting-library commit. All chapters share
one local `.venv`.

## Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. Use
PowerShell to run:

```powershell
git clone https://github.com/lorendw7/Deep-Learning-For-Biology.git
cd Deep-Learning-For-Biology
uv sync --locked
```

uv creates `.venv`, installs the saved package versions, and downloads Python 3.11
if needed. This Windows environment does not use Conda. `uv run` uses `.venv`
automatically, so manual activation is optional.

TensorFlow needs the [Microsoft Visual C++ runtime](https://www.tensorflow.org/install/pip#windows-native).
It is already available on the Windows computer used to verify this setup.

## Use PyCharm

Open the repository folder. In **Settings > Python > Interpreter**, select the
existing interpreter:

```text
D:\CS\Code\Deep-Learning-For-Biology\.venv\Scripts\python.exe
```

If this project was open before the environment update, refresh the interpreter
or reopen the project. It should now show Python 3.11.

Start chapter scripts with the local setup call:

```python
from book_runtime import configure

configure()

import jax
import dlfb

print(jax.devices())
```

Call `configure()` before importing JAX, Flax, Transformers, or `dlfb`. It loads
TensorFlow first, which avoids a Windows DLL failure seen when JAX is imported
first. It also selects the CPU backend and sets local data and model-cache paths.
If JAX was already imported, restart the script or notebook kernel first.

Run scripts from PyCharm or the repository terminal:

```powershell
uv run --locked python chapters/chapter-01/hello.py
```

## Run the official notebooks locally

Create Windows copies of the official notebooks, then start JupyterLab:

```powershell
uv run --locked python scripts/prepare_notebooks.py
uv run --locked jupyter lab
```

Open a file under `notebooks/` and use the Python kernel from this repository's
`.venv`. The first code cell checks the interpreter and sets up the local runtime.

The preparation script replaces Colab installation, authentication, and download
cells with local setup and data-download instructions. It also clears saved
outputs. Existing local notebooks are kept. Use `--force` only when you want to
overwrite them. Generated copies are ignored by Git; keep your own study work
under `chapters/`.

Download the data for the chapter you are studying. For example:

```powershell
uv run --locked dlfb-provision --chapter proteins --destination data/assets
```

Chapter names are `proteins`, `dna`, `graphs`, `cancer`, and `localization`.
Downloads can be large. Add `--no-models` to download only the datasets. The
notebooks can still need pretrained model files later.

Local paths:

- `data/assets/`: chapter datasets and saved book models.
- `data/huggingface/`: downloaded Hugging Face models.
- `outputs/`: generated results.

These folders are ignored by Git. `book_runtime.configure()` sets `ASSETS_DIR`
for the supporting library. It respects an existing `ASSETS_DIR` or `HF_HOME` if
you choose your own paths.

## Verify the environment

Run:

```powershell
uv pip check
uv run --locked python scripts/check_environment.py
```

The check uses real `dlfb` code and small synthetic inputs. It checks package
versions and imports, trains the protein model for three steps, runs one training
step for each remaining chapter, tests a small ESM model and torchvision,
checks UMAP, saves and restores a checkpoint, and executes a real Jupyter kernel.

The official chapter 5 module loads ResNet-18, ResNet-34, and ResNet-50 on import.
The first check downloads their weights into `data/huggingface/`. Later runs use
the cache. The ESM check creates a small random model and does not download the
large protein language model.

The upstream ResNet-18/34 Flax checkpoints produce missing/unused-weight warnings
with the official Transformers version. The chapter 5 check trains `SimpleCnn`;
it does not validate those pretrained checkpoints for transfer learning.

These checks verify the local runtime. They do not run full chapter datasets,
long training jobs, or assess pretrained-model accuracy. CPU training on the full
book projects can be slow.

See the [verification record](docs/environment-verification.md) for the checks
run on October 19, 2026.

## Windows changes

- Linux Triton and JAX CUDA plugins are omitted. Native Windows uses CPU JAX;
  use the official Colab notebooks or a separate WSL2/Linux environment for GPU
  work. See the [JAX platform support table](https://docs.jax.dev/en/latest/installation.html#supported-platforms).
- Official type-stub packages are omitted because they are for static type
  checking, not running the chapters.
- PyArrow 19.0.1 keeps the API used by Datasets 2.19.1. Numba 0.61.2 supports
  NumPy 2.1.2. TensorFlow Metadata 1.16.1 keeps the TensorFlow/Protobuf stack
  compatible. SciPy is pinned to 1.15.3.
- JupyterLab 4.3.5, ipykernel 6.29.5, and IPython 8.18.1 work with the official
  Traitlets 5.7.1 constraint. Setuptools 75.8.2 keeps the older OGB dependency
  working.
- The local startup call handles TensorFlow/JAX import order. The supporting
  library itself is installed from its unchanged upstream commit.

## Study work

Add chapter code and short English notes under folders such as
`chapters/chapter-01/`. Explain how to run each example and record its data source.
Use `uv add package-name` when new work needs another package, then run the checks
again and commit both `pyproject.toml` and `uv.lock`.

## License

Original work in this repository uses the [MIT License](LICENSE). The official
`dlfb` library and notebooks use the Apache 2.0 License. The book, datasets, and
model weights keep their own licenses. Keep source links and license information
when using material from elsewhere.
