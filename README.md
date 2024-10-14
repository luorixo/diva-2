## DIVA Experiment

Detecting InVisible Attacks (DIVA) is a framework for detecting poisoned datasets using meta-learning techniques and dataset complexity measures. This repository contains the full codebase, experiments, and visualisations developed to evaluate the transferability and robustness of DIVA in detecting various types of data poisoning attacks.

This project explores the capabilities of the DIVA framework in detecting poisoning attacks under agnostic conditions. DIVA aims to generalise across various poisoning strategies without prior knowledge of attack types, leveraging meta-learning and dataset complexity measures.

The primary objectives include:

- Assessing DIVA’s performance across different datasets and machine learning models.
- Evaluating its ability to detect previously unseen poisoning attacks.
- Investigating its effectiveness against clean-label backdoor attacks (CLBA).

This codebase contains the initial steps of the pipeline used to generate the meta-database, and the test meta-databases used for the final meta-learners.

## Installation
To get started, clone this repository and install the necessary dependencies in `requirements.txt`. You can set this up in an `venv`.



To run the pipeline in order to generate your training meta-database, you can run:

```
python .\metadb_generation.py -n Number of datasets. -f Your output file. -s Spacing between poisoning rates. -m Max poisoning.
```

This same command can be run to generate a test meta-database. 