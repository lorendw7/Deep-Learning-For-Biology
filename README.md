# Deep Learning for Biology

This is my personal study repository for *Deep Learning for Biology: Harness AI
to Solve Real-World Biology Problems* by Charles Ravarani and Natasha Latysheva
(O'Reilly).

I use this repository to work through the book, write code, and keep short notes.
All documentation and study notes use plain, simple English.

## Current progress

The Python environment is ready. Chapter code and study notes will be added as I
read the book.

## Set up the environment

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). Install uv first if it is not
already on your computer. Run these commands from a terminal:

```sh
git clone https://github.com/lorendw7/Deep-Learning-For-Biology.git
cd Deep-Learning-For-Biology
uv sync --locked
```

uv creates a local `.venv` folder and installs the versions in `uv.lock`. It can
download Python 3.13 if needed. The virtual environment stays on your computer
and is not stored in Git.

The starting environment includes:

- JupyterLab and ipykernel for notebooks.
- NumPy, pandas, and SciPy for data and calculations.
- Matplotlib for plots.
- scikit-learn for machine learning.
- PyTorch for deep learning.
- Biopython for biological data.

This setup is for local study. It does not include a separate CUDA setup. Add
chapter-specific tools when an example needs them.

## Run code and notebooks

Start JupyterLab:

```sh
uv run --locked jupyter lab
```

Use the Python kernel from `.venv`. In a code editor, select
`.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on macOS and Linux.

Check that Python and PyTorch work:

```sh
uv run --locked python -c "import sys, torch; print(sys.version); print(torch.ones(2, 2))"
```

Run a Python file with `uv run --locked python path/to/file.py`.

You do not need to activate the environment when using `uv run`. To activate it
manually in Windows PowerShell, run:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS and Linux, run:

```sh
source .venv/bin/activate
```

## Organize study work

Add chapter work under folders such as `chapters/chapter-01/`. Keep each chapter's
code, notebooks, and a short `README.md` together. Explain what the example does,
how to run it, and what I learned.

Keep downloaded datasets in `data/` and generated results in `outputs/`. These
folders are ignored by Git. Record the data source and download steps in the
chapter notes so an example can be run again.

## Add dependencies

When a chapter needs another Python package, run:

```sh
uv add package-name
```

Commit both `pyproject.toml` and `uv.lock` after changing dependencies. To recreate
the saved environment later, run `uv sync --locked`.

## License

Original work in this repository uses the [MIT License](LICENSE). The book and
any third-party code or datasets keep their own licenses. Add source links and
license information when using material from elsewhere.
