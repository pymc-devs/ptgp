Contributing
============

Development install
-------------------

.. code-block:: bash

    git clone https://github.com/pymc-devs/ptgp.git
    cd ptgp
    pixi install
    pixi run pre-commit install

Running tests
-------------

ptgp uses pytest:

.. code-block:: bash

    pixi run test

``tests/test_cubic_floor.py`` (``pixi run cubic-floor``) checks that each
model's training graph compiles to the minimum number of O(N³)
factorizations. Run it after touching ``ptgp/rewrites.py``,
``ptgp/objectives.py``, or kernel evaluation.

Pre-commit hooks
----------------

``pixi run pre-commit install`` runs these on every commit;
``pixi run hooks`` runs them on all files. CI runs the same set.

* ``ruff`` (lint, with fixes) and ``ruff-format``, line length 100.
  Notebooks are excluded from ``ruff-format``.
* ``no-print-statements`` in ``ptgp/``.
* ``no-commit-to-branch``: commits directly to ``main`` are rejected.
* ``check-kernel-gallery``: every kernel exported from ``ptgp.kernels``
  needs a gallery cover or a blacklist entry. See
  :ref:`adding-a-gallery-kernel`.
* Whitespace, end-of-file, merge-conflict, TOML/YAML syntax, and leftover
  debugger checks.

Branches and pull requests
--------------------------

Work on a feature branch and open a pull request against ``main``. CI runs
the test suite, pre-commit, and mypy on every PR. PRs that touch the docs
also get a Read the Docs preview build.
