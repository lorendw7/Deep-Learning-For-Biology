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

## ResNet loading and inference

The ResNet-18/34 loading warnings were fixed by matching the checkpoint's basic
block names to Flax's names before tensor conversion. The installed upstream
library and Transformers version are unchanged. The local adapter checks complete
backbone coverage and fails on missing, extra, wrong-shaped, or non-finite tensors.

`uv run --locked python scripts/check_pretrained_resnets.py` passed. It captures
Transformers loading warnings without changing their log level or filtering them;
none were emitted. The original PyTorch classifier also loaded with no missing,
unexpected, or mismatched keys.

The same detailed check passed again with `--offline`. The full environment
check, including all five chapters and the Jupyter kernel, also passed after
enabling the new loader.

| Check | ResNet-18 | ResNet-34 |
| --- | --- | --- |
| Backbone weight/state tensors matched exactly | 100 | 180 |
| Maximum feature-map absolute error | 2.575e-5 | 5.150e-5 |
| Maximum ImageNet logit absolute error | 6.199e-6 | 5.245e-6 |
| Top-5 predictions match PyTorch | Yes | Yes |
| Sample photo's top label | tabby cat | tabby cat |
| Book model inference output shape | `(1, 2)` | `(1, 2)` |

The comparison uses a sample cat photo from
[Hugging Face's documentation images](https://huggingface.co/datasets/huggingface/documentation-images/blob/2dd75e791509da38222ecbd3bfaa2b44ac658ef4/coco_sample.png)
and fixed random input, each at 224 by 224 pixels. Feature maps and reconstructed
ImageNet logits match the independent PyTorch execution within `rtol=1e-4` and
`atol=1e-4`. Every backbone tensor and BatchNorm state is compared exactly to the
original PyTorch tensor. The ImageNet head and PyTorch-only batch counters are
deliberately excluded from the book backbone.

The book's `FinetunedResNet` received the exact loaded weights and BatchNorm
states and produced finite inference output for both depths. Negative checks
confirmed that incomplete or corrupt states are rejected instead of repaired
with random weights. The generated JSON report stays under ignored `outputs/`.

## Scope

Model checks use small synthetic inputs. Full datasets and long training runs
were not executed. A small randomly initialized ESM model was used instead of
downloading the large protein language model.

ResNet inference and weight transfer are verified as described above. The
skin-lesion classification head is newly initialized by the book and needs
training. These checks do not establish medical prediction quality or a fully
trained skin-cancer classifier.

Windows changes and exact package sources are described in the root README.
