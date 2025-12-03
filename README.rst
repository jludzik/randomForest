.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

===============================================
Random Forest Classifier: Implementation & Analysis
===============================================

This project implements a Random Forest classifier from scratch using NumPy and investigates its performance compared to the industry-standard Scikit-Learn implementation. It was developed as part of the "Programming in Python Language" course at AGH University of Krakow.

Project Overview
================

The primary objective is to build a fully functional machine learning algorithm to understand its mathematical foundations. The project focuses on:

1.  **Implementation:** Creating ``DecisionTree`` and ``RandomForest`` classes using only low-level array operations (NumPy).
2.  **Parity Analysis:** Benchmarking the custom implementation against ``sklearn.ensemble.RandomForestClassifier`` on real datasets (Wine Quality).
3.  **Software Engineering:** Utilizing modern Python tooling like PyScaffold, Sphinx, and Pytest.

Methodology
===========

Algorithm Implementation
------------------------
The core logic is divided into two main components:

* **Decision Tree:** Implements the ID3/CART-like algorithm. It uses **Shannon Entropy** and **Information Gain** to recursively split data nodes until a leaf is formed.
* **Random Forest (Ensemble):** Implements **Bagging (Bootstrap Aggregation)**. Multiple trees are trained on random subsets of data (with replacement), and the final prediction is determined by a majority vote.

Technology Stack
----------------
* **Core:** Python 3.13, NumPy (Vectorized calculations).
* **Data Handling:** Pandas.
* **Visualization:** Matplotlib.
* **Testing & Docs:** Pytest, Sphinx.

Project Structure
=================

The codebase is organized into a modular structure using PyScaffold:

.. code-block:: text

    ├── src/
    │   └── randomforest/
    │       ├── DecisionTree.py      # Node and DecisionTree classes (Entropy, Split logic)
    │       └── RandomForest.py      # RandomForest class (Bootstrapping, Aggregation)
    ├── tests/
    │   └── test_random_forest.py    # Unit tests for algorithm correctness
    ├── example/
    │   ├── compare_plots.py         # Main script: Benchmarks vs Scikit-Learn
    │   └── winequality-red.csv      # Dataset used for experiments
    ├── docs/                        # Sphinx documentation source files
    ├── requirements.txt             # Project dependencies
    └── setup.cfg                    # Package configuration

Setup and Installation
======================

To set up the development environment, ensure you have Python 3.8+ installed.

1.  **Clone the repository:**

    .. code-block:: bash

        git clone https://github.com/jludzik/randomForest
        cd randomForest

2.  **Create and activate a virtual environment:**

    .. code-block:: bash

        python -m venv .venv

        # Windows (PowerShell):
        .venv\Scripts\activate

        # macOS / Linux:
        source .venv/bin/activate

3.  **Install dependencies and the package:**

    .. code-block:: bash

        pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
        pip install pytest pytest-cov sphinx

Running the Experiment
======================

1.  **Verify Correctness (Unit Tests):**
    Run the test suite to ensure the entropy calculations and tree splits are working correctly.

    .. code-block:: bash

        pytest

2.  **Run Comparative Analysis:**
    Execute the main experiment script. This will train both the custom model and Scikit-Learn model on the Wine dataset and generate performance plots.

    .. code-block:: bash

        python example/compare_plots.py

3.  **Generate Documentation:**
    Build the API documentation locally.

    .. code-block:: bash

        cd docs
        make html
        # Open docs/_build/html/index.html to view

Results
=======

The experiment generates a comparison of Accuracy vs. Number of Trees and Tree Depth. The artifacts are saved in the ``example/`` directory.

* **Visualization (`comparision_plots.png`):** Shows that the custom implementation achieves accuracy parity with Scikit-Learn, converging as the number of estimators increases.

.. image:: https://raw.githubusercontent.com/jludzik/randomForest/main/example/comparision_plots.png
   :width: 800
   :alt: Comparison of Accuracy results

Authors and Context
===================

* **Authors:**
    * Jakub Łudzik
    * Filip Żurek
* **Institution:** AGH University of Krakow
* **Faculty:** Faculty of Computer Science, Electronics and Telecommunications
* **Field of Study:** Electronics and Telecommunications
* **Course:** Programming in Python Language

License
=======

This software is distributed under the MIT License. Refer to the `LICENSE.txt` file for the full text.

--------------------------------------------------------------------------------
*AGH University of Krakow - Programming in Python Language Course Project 2025*