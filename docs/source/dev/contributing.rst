Contributing
============

.. note::

   **WRITEME.** This page is a stub. Cover the pre-commit hooks, branching /
   PR workflow, and where new kernels and likelihoods should live.

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
