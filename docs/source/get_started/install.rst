Installation
============

.. note::

   **WRITEME.** This page is a stub. Flesh out with full install instructions,
   supported Python versions, and any platform-specific notes.

ptgp targets Python ``>= 3.12`` and depends on the conda-forge / PyPI
releases of PyMC (``6.3.x``), PyTensor (``3.3.x``) and pytensor-ml
(``0.3.x``); see ``pyproject.toml`` for the exact pins.

From PyPI
---------

ptgp and all of its dependencies ship wheels on PyPI:

.. code-block:: bash

    pip install ptgp

PyTensor's C backend needs a C++ compiler on your PATH (Xcode command line
tools on macOS, ``g++`` on Linux). On Windows, prefer the pixi or conda
routes below, which bring a compiler along.

From source
-----------

.. code-block:: bash

    pip install git+https://github.com/pymc-devs/ptgp.git

With conda
----------

``conda_envs/environment.yaml`` is the full development environment as a
conda env file (ptgp is installed in editable mode from the checkout):

.. code-block:: bash

    git clone https://github.com/pymc-devs/ptgp.git
    cd ptgp
    conda env create -f conda_envs/environment.yaml
    conda activate ptgp

Development install
-------------------

The repo ships a `pixi <https://pixi.sh>`_ manifest (``[tool.pixi.*]`` in
``pyproject.toml``, locked in ``pixi.lock``) that builds the full dev
environment, with ptgp installed in editable mode:

.. code-block:: bash

    git clone https://github.com/pymc-devs/ptgp.git
    cd ptgp
    pixi install
    pixi run test            # also: lint, mypy, docs, kernel (Jupyter kernel)
    pixi run pre-commit install

On Apple silicon the environment also includes `MLX <https://ml-explore.github.io/mlx/>`_
for PyTensor's ``MLX`` compile mode (float32 only).

See :doc:`/dev/contributing` for the rest of the contributor setup.
