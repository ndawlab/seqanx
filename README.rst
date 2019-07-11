.. image:: https://img.shields.io/travis/ndawlab/seqanx.svg
        :target: https://travis-ci.org/ndawlab/seqanx

.. image:: https://img.shields.io/badge/python-3.6-blue.svg
        :target: https://www.python.org/downloads/release/python-360/

.. image:: https://img.shields.io/github/license/mashape/apistatus.svg
        :target: https://github.com/ndawlab/seqanx/blob/master/LICENSE

Sequential Choice in Anxiety
============================

Anxiety disorders are characterized by a range of aberrations in the processing and response to threat, but there is little clarity what core pathogenesis might underlie these symptoms. Here we propose a decision theoretic analysis of maladaptive avoidance and embody it in a reinforcement learning model, which shows how a localized bias in beliefs can formally explain a range of phenomena related to anxiety. The core observation, implicit in standard decision theoretic accounts of sequential evaluation, is that avoidance should be protective: if danger can be avoided later, it poses no threat now. We show how a violation of this assumption - a pessimistic, false belief that later avoidance will be unsuccessful - leads to a characteristic propagation of fear and avoidance to situations far antecedent of threat. This single deviation can explain a surprising range of features of anxious behavior, including exaggerated threat appraisals, fear generalization, and persistent avoidance.

Author
^^^^^^
Sam Zorowitz (zorowitz [at] princeton.edu)

Project Organization
^^^^^^^^^^^^^^^^^^^^
::

    ├── figures                      <- Figures for presentations & manuscript.
    │   
    ├── manuscripts                  <- Manuscripts & RLDM 2019 abstract/poster.
    │   
    ├── notebooks                    <- Analysis notebooks for the projects.
    │   ├── 01_OpenField.ipynb       <- Demo of pessimistic RL in toy MDP.
    │   ├── 02_AppAvo.ipynb          <- Model of approach-avoidance bias.
    │   ├── 03_Helplessness.ipynb    <- Model of anxiety-to-depression transition.
    │   ├── 04_DecisionTree.ipynb    <- Model of aversive pruning. 
    │   ├── 05_FreeChoice.ipynb      <- Model of free choice premium.
    │   ├── 06_CliffWalking.ipynb    <- Reproduction of Gaskett (2003).
    │   
    ├── sisyphus                     <- Source code used in notebooks (installation instructions below).
    │   ├── envs                     <- Task environments.
    │   ├── mdp                      <- RL algorithms (value iteration, temporal difference learning).
    │   ├── tests                    <- Continuous integration tests.
    │   
    ├── requirements.txt             <- Python packages used in this project.

Installation
^^^^^^^^^^^^

This repository hosts the :code:`sisyphus` package, which is used in all of this project's simulations. The package contains functions for constructing arbitrary MDP environments and several reinforcement learning algorithms.

To install the code through Github, open a terminal and run:

.. code-block:: bash

    pip install git+https://github.com/ndawlab/seqanx.git

Alternately, you can clone the repository and install locally:

.. code-block:: bash

    git clone https://github.com/ndawlab/seqanx
    cd seqanx
    pip install -e .

Once installed, the simulations (found in the notebooks folder) should be reproducible on any computer.
