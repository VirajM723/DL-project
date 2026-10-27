"""Test harness: execute the notebook's code cells (up to the first training cell) and expose its namespace.

The notebook is the single source of truth, so the tests import its functions by running its cells
with the 1-minute smoke configuration, inside a temporary working directory (figures and results
written by those cells never touch the repository).
"""
import ast
import os
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "TrackNetV5_TinyObject_Tracking.ipynb"
STOP_MARKER = "MODELS, HIST = {}, {}"          # first cell that trains models


def load_notebook_namespace(workdir, profile="cpu"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nbformat

    os.environ["TN_SMOKE"] = "1"
    os.environ["TN_PROFILE"] = profile
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns = {"__name__": "tracknet_notebook"}
    old = os.getcwd()
    os.chdir(workdir)
    try:
        stopped = False
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            if STOP_MARKER in cell.source:
                stopped = True
            # before the first training cell run everything; after it, only pure definition cells
            if not stopped or _definitions_only(cell.source):
                exec(compile(cell.source, f"<notebook cell {i}>", "exec"), ns)
                plt.close("all")
        if not stopped:
            raise RuntimeError(f"stop marker {STOP_MARKER!r} not found in the notebook")
    finally:
        os.chdir(old)
    return types.SimpleNamespace(**ns)


def _definitions_only(source):
    tree = ast.parse(source)
    return bool(tree.body) and all(isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
                                   for n in tree.body)


@pytest.fixture(scope="session")
def nb(tmp_path_factory):
    return load_notebook_namespace(tmp_path_factory.mktemp("nb_cpu"))
