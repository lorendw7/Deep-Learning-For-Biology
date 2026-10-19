"""Create local Windows copies of the pinned official notebooks."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c705571f6041a97c45fefbb2490c4a83e81a5672"
CHAPTERS = {2: "proteins", 3: "dna", 4: "graphs", 5: "cancer", 6: "localization"}


def adapt(notebook, chapter, name):
    setup = (
        "from pathlib import Path\n"
        "import sys\n"
        "import book_runtime\n\n"
        "ROOT = next(path for path in (Path.cwd(), *Path.cwd().parents)\n"
        "            if (path / 'pyproject.toml').is_file())\n"
        "assert Path(sys.prefix).resolve() == (ROOT / '.venv').resolve(), 'Select the project .venv kernel.'\n"
        "book_runtime.configure(ROOT)\n"
        "print('Python:', sys.executable)\n"
        "print('Assets:', ROOT / 'data' / 'assets')\n"
    )
    replaced = {"clone": 0, "install": 0, "download": 0}
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        if "!git clone" in source:
            source = setup
            replaced["clone"] += 1
        elif "uv pip install" in source:
            cell["cell_type"] = "markdown"
            source = "Dependencies are installed with `uv sync --locked` in the repository root."
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
            replaced["install"] += 1
        elif "!dlfb-provision" in source:
            cell["cell_type"] = "markdown"
            source = (
                "Download this chapter's data from a terminal in the repository root before continuing:\n\n"
                f"```powershell\nuv run --locked dlfb-provision --chapter {name} --destination data/assets\n```\n\n"
                "This can download large files. Add `--no-models` to download only the datasets."
            )
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
            replaced["download"] += 1
        else:
            source = source.replace("%env JAX_DISABLE_JIT=False", "import os\nos.environ['JAX_DISABLE_JIT'] = 'False'")
        cell["source"] = source.splitlines(keepends=True)
    if replaced != {"clone": 1, "install": 1, "download": 1}:
        raise ValueError(f"Unexpected upstream setup cells: {replaced}")
    notebook["cells"].insert(0, {
        "cell_type": "markdown", "metadata": {},
        "source": [
            f"# Chapter {chapter}: local Windows copy\n",
            "Code source: https://github.com/deep-learning-for-biology/notebooks\n\n",
            f"Upstream revision: `{REVISION}`. Upstream code uses the Apache 2.0 License.\n\n",
            "Local changes: project environment, local asset paths, and manual data downloads.\n",
            "Use the Python kernel from this repository's `.venv`.\n",
        ],
    })
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (book .venv)", "language": "python", "name": "python3"
    }
    notebook["metadata"].pop("colab", None)
    notebook["metadata"]["language_info"]["version"] = "3.11"
    notebook["nbformat_minor"] = max(notebook.get("nbformat_minor", 0), 5)
    for index, cell in enumerate(notebook["cells"]):
        cell.setdefault("id", hashlib.sha256(f"{name}:{index}".encode()).hexdigest()[:12])
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated notebooks.")
    args = parser.parse_args()
    output = ROOT / "notebooks"
    output.mkdir(exist_ok=True)
    for chapter, name in CHAPTERS.items():
        filename = f"chapter_{chapter}_{name}.ipynb"
        target = output / filename
        if target.exists() and not args.force:
            print(f"Kept existing {target.relative_to(ROOT)}")
            continue
        url = f"https://raw.githubusercontent.com/deep-learning-for-biology/notebooks/{REVISION}/notebooks/{filename}"
        with urlopen(url, timeout=60) as response:
            notebook = json.load(response)
        notebook = adapt(notebook, chapter, name)
        target.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        print(f"Created {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
