.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

============
RandomForest
============

Final project for the **Python Programming** course.
This package contains a custom implementation of the **Random Forest** and **Decision Tree** algorithms, built from scratch using NumPy, along with a performance comparison against the Scikit-Learn library.

Authors and Context
===================

* **Project Authors (Developers):**
    * Jakub Łudzik
    * Filip Żurek
* **Peer Reviewers (Testers):**
    * Paweł Michalcewicz
    * Krystian Ruszczak
* **Institution:** AGH University of Krakow
* **Faculty:** Faculty of Computer Science, Electronics and Telecommunications
* **Field of Study:** Electronics and Telecommunications
* **Course:** Programming in Python Language

Project Description
===================
The goal of this project was to create a fully functional Random Forest classifier that:
* Implements the Decision Tree algorithm (tree growth, entropy calculation, information gain).
* Implements the Random Forest ensemble method (bootstrapping, vote aggregation).
* Adheres to the Scikit-Learn interface (providing ``fit`` and ``predict`` methods).

Installation Guide (Step-by-Step)
=================================
Follow these steps to set up the project environment from scratch.

1. **Prerequisites**
   Ensure you have Python 3.8+ installed. You can check this by running:

   .. code-block:: bash

       python --version

2. **Create a Virtual Environment**
   It is highly recommended to run this project in an isolated virtual environment. Run the following command in the project root directory:

   .. code-block:: bash

       python -m venv .venv

3. **Activate the Virtual Environment**
   You need to activate the environment before installing packages.

   * **On macOS / Linux:**

     .. code-block:: bash

         source .venv/bin/activate

   * **On Windows (PowerShell/CMD):**

     .. code-block:: bash

         .venv\Scripts\activate

   *(You should see `(.venv)` appear at the beginning of your terminal line).*

4. **Install the Package and Dependencies**
   Install the project in editable mode along with required libraries for plotting and testing:

   .. code-block:: bash

       pip install --upgrade pip
       pip install -e .
       pip install matplotlib pandas scikit-learn pytest pytest-cov sphinx

Usage
=====

Ensure your virtual environment is activated (step 3 above) before running these commands.

1. **Unit Tests:**
   To verify the correctness of the implementation:

   .. code-block:: bash

       pytest

2. **Comparison with Scikit-Learn (Plots):**
   To generate comparison plots (Accuracy vs. Tree Depth / Number of Estimators):

   .. code-block:: bash

       python example/compare_plots.py

   The results will be saved as ``comparision_plots.png``.

    .. image:: https://github.com/jludzik/randomForest/blob/main/example/comparision_plots.png?raw=true
       :width: 800
       :alt: Comparison of Accuracy results

3. **Documentation:**
   The documentation source is located in the ``docs/`` directory. To build the HTML version:

   .. code-block:: bash

       cd docs
       make html
       # Open docs/_build/html/index.html in your browser

License
=======

This software is distributed under the MIT License. Refer to the `LICENSE.txt` file for the full text.

---
*AGH University of Krakow - Programming in Python Language Course Project 2025*
