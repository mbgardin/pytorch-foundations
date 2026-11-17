"""Execute each notebook in a fresh kernel without modifying notebook files.

Run from the repository root with: uv run python scripts/check_notebooks.py
Notebook 07 writes its example checkpoint to the ignored artifacts directory.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No notebooks found.")
    for path in notebooks:
        print(f"Running {path.name} ...", flush=True)
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(
            notebook, timeout=180, kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        client.execute()
    print(f"All {len(notebooks)} notebooks completed successfully.")


if __name__ == "__main__":
    main()
