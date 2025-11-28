.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

============
RandomForest
============

Final project for the **Python Programming** course.
This package contains a custom implementation of the **Random Forest** and **Decision Tree** algorithms, built from scratch using NumPy, along with a performance comparison against the Scikit-Learn library.

Authors
=======
* Jakub Łudzik
* Filip Żurek

Project Description
===================
The goal of this project was to create a fully functional Random Forest classifier that:
* Implements the Decision Tree algorithm (tree growth, entropy calculation, information gain).
* Implements the Random Forest ensemble method (bootstrapping, vote aggregation).
* Adheres to the Scikit-Learn interface (providing ``fit`` and ``predict`` methods).

Requirements and Installation
=============================
The project relies on ``numpy``, ``pandas``, ``matplotlib``, and ``scikit-learn`` (for benchmarking purposes).

To install the package in development mode:

.. code-block:: bash

    pip install -e .

Usage
=====

1. **Unit Tests:**
   To verify the correctness of the implementation:

   .. code-block:: bash

       pytest

2. **Comparison with Scikit-Learn (Plots):**
   To generate comparison plots (Accuracy vs. Tree Depth / Number of Estimators):

   .. code-block:: bash

       python example/compare_plots.py

   The results will be saved as ``comparision_plots.png``.

    .. image:: example/comparision_plots.png
       :width: 800
       :alt: Comparison of Accuracy results

3. **Documentation:**
   The documentation source is located in the ``docs/`` directory. To build the HTML version:

   .. code-block:: bash

       cd docs
       make html