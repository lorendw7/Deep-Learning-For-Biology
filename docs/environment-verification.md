# Environment verification

Date: October 19, 2026.

Platform: native Windows x64, Python 3.11.15, CPU backends. The final checks used
this repository's `.venv`, after recreating it from `uv.lock`.

## Results

- `uv sync --locked`: passed.
- `uv pip check`: all 215 installed packages are compatible.
- `uv run --locked python scripts/check_environment.py`: passed.
- All 63 modules in the pinned `dlfb` library imported.
- Chapter 2: TensorFlow data batches and three official protein-model training
  steps passed. Model parameters changed and losses were finite.
- Chapters 3 to 6: the official DNA, graph, simple image CNN, and localization
  models each completed a training step with finite losses.
- UMAP, a small ESM forward/backward pass, and the torchvision NMS operation passed.
- The official Orbax save/restore functions preserved model parameters and metadata.
- A real Jupyter kernel ran the local setup and a JAX calculation using `.venv`.
- All five generated official notebooks passed format and Python syntax checks.
  Their setup cells and import statements ran successfully in the project kernel.
- `dlfb-provision --help` worked. The official dataset index was reachable.
- The existing chapter 1 `hello.py` still ran and reported PyTorch 2.6.0+cpu.

## Scope

Model checks use small synthetic inputs. Full datasets and long training runs
were not executed. A small randomly initialized ESM model was used instead of
downloading the large protein language model.

The chapter 5 import downloaded the upstream ResNet weights. ResNet-18/34 Flax
checkpoints emitted missing/unused-weight warnings with Transformers 4.49.0.
The chapter 5 training check uses `SimpleCnn`; it does not certify pretrained
ResNet transfer learning or prediction quality.

Windows changes and exact package sources are described in the root README.
