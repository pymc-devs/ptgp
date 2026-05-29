Installation
============

.. note::

   **WRITEME.** This page is a stub. Flesh out with full install instructions,
   supported Python versions, and any platform-specific notes.

ptgp targets Python ``>= 3.12``. It depends on PyMC (``>= 6.0``) and on
PyTensor's ``main`` branch.

From PyPI
---------

.. code-block:: bash

    pip install ptgp

.. note::

   ptgp is not yet released on PyPI. Until then, install from source.

From source
-----------

.. code-block:: bash

    git clone https://github.com/bwengals/ptgp.git
    cd ptgp
    pip install -e .

PyTensor and PyMC
-----------------

ptgp currently requires the development versions of PyTensor (``main``) and
PyMC (``>= 6.0``). The pinned versions in the project's conda environments
under ``conda_envs/`` are the reference setup the maintainer uses; mirror
those if your install is misbehaving.

.. code-block:: bash

    pip install git+https://github.com/pymc-devs/pytensor@main
    pip install pymc

Development install
-------------------

.. code-block:: bash

    git clone https://github.com/bwengals/ptgp.git
    cd ptgp
    pip install -e ".[dev]"
    pre-commit install

See :doc:`/dev/contributing` for the rest of the contributor setup.
