# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import re
import pytest
import html
import os

_nbdir = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks')

_nbsubdirs = ['.', 'CustomerScenarios', 'Solutions']  # TODO: add AutoML notebooks

# filter directories by regex if the NOTEBOOK_DIR_PATTERN environment variable is set
_nbsubdirs = [d for d in _nbsubdirs if re.match(os.getenv('NOTEBOOK_DIR_PATTERN', '.*'), d)]

_notebooks = [
    os.path.join(subdir, path) for subdir
    in _nbsubdirs for path in os.listdir(os.path.join(_nbdir, subdir)) if
    path.endswith('.ipynb')]
# omit the lalonde notebook
_notebooks = [nb for nb in _notebooks if "Lalonde" not in nb]


@pytest.mark.parametrize("file", _notebooks)
@pytest.mark.notebook
def test_notebook(file):
    import nbformat
    import nbconvert

    nb = nbformat.read(os.path.join(_nbdir, file), as_version=4)
    # require all cells to complete within 15 minutes, which will help prevent us from
    # creating notebooks that are annoying for our users to actually run themselves
    ep = nbconvert.preprocessors.ExecutePreprocessor(
        timeout=1800, allow_errors=True, extra_arguments=["--HistoryManager.enabled=False"])

    ep.preprocess(nb, {'metadata': {'path': _nbdir}})
    errors = [nbconvert.preprocessors.CellExecutionError.from_cell_and_msg(cell, output)
              for cell in nb.cells if "outputs" in cell
              for output in cell["outputs"]
              if output.output_type == "error"]
    if errors:
        err_str = "\n".join(html.unescape(str(err)) for err in errors)
        raise AssertionError("Encountered {0} exception(s):\n{1}".format(len(errors), err_str))
