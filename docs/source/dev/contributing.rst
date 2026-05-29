Contributing
============

.. note::

   **WRITEME.** This page is a stub. Cover environment setup, the
   pre-commit hooks, how to run tests, the GPJax-as-reference testing
   convention, branching / PR workflow, and where new kernels and
   likelihoods should live.

Development install
-------------------

.. code-block:: bash

    git clone https://github.com/bwengals/ptgp.git
    cd ptgp
    pip install -e ".[dev]"
    pre-commit install

Running tests
-------------

ptgp uses pytest:

.. code-block:: bash

    python -m pytest tests/

Many tests compare against GPJax (the reference implementation) at
``atol=1e-5``. GPJax runs in float32 and ptgp in float64; that tolerance
is the cross-library cap.
