.. image:: https://img.shields.io/badge/python-3.6-blue.svg
        :target: https://www.python.org/downloads/release/python-360/

.. image:: https://img.shields.io/github/license/mashape/apistatus.svg
        :target: https://github.com/ndawlab/seqanx/blob/master/LICENSE

Sequential Choice in Anxiety
============================

Anxiety disorders are characterized by a range of aberrations in the processing and response to threat, but there is little clarity what core pathogenesis might underlie these symptoms. Here we propose a decision theoretic analysis of maladaptive avoidance and embody it in a reinforcement learning model, which shows how a localized bias in beliefs can formally explain a range of phenomena related to anxiety. The core observation, implicit in standard decision theoretic accounts of sequential evaluation, is that avoidance should be protective: if danger can be avoided later, it poses no threat now. We show how a violation of this assumption --- a pessimistic, false belief that later avoidance will be unsuccessful --- leads to a characteristic propagation of fear and avoidance to situations far antecedent of threat. This single deviation can explain a surprising range of features of anxious behavior, including exaggerated threat appraisals, fear generalization, and persistent avoidance. 

Organization
^^^^^^^^^^^^

This repository contains all of the analysis code to reproduce the results associated with this project. 


Installation
^^^^^^^^^^^^



To install the code used to generate the results in the manuscript, open a terminal and run:

.. code-block:: bash

    pip install git+https://github.com/ndawlab/seqanx.git

Alternately, you can clone the repository and install locally:

.. code-block:: bash

    git clone https://github.com/ndawlab/seqanx
    cd seqanx
    pip install -e .
