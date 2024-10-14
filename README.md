# DIVA Experiment

Detecting InVisible Attacks (DIVA) is a framework for detecting poisoned datasets using meta-learning techniques and dataset complexity measures. This repository contains the full codebase, experiments, and visualisations developed to evaluate the transferability and robustness of DIVA in detecting various types of data poisoning attacks.

This project explores the capabilities of the DIVA framework in detecting poisoning attacks under agnostic conditions. DIVA aims to generalise across various poisoning strategies without prior knowledge of attack types, leveraging meta-learning and dataset complexity measures.

The primary objectives include:

- Assessing DIVA’s performance across different datasets and machine learning models.
- Evaluating its ability to detect previously unseen poisoning attacks.
- Investigating its effectiveness against clean-label backdoor attacks (CLBA).

This codebase contains the initial steps of the pipeline used to generate the meta-database, and the test meta-databases used for the final meta-learners.

# Files Contained

The main repository used for our experiments is `diva`, which is where the meta-databases and meta-learners used in our experiments are stored. `project` was our old abandoned pipeline that followed the original DIVA more closely, but we decided to create a new pipeline.

In `diva`, it contains the scripts for each poisoner, inside the `scripts` directory, which contains the implementation of every poisoner we used, except for Clean-label Backdoor attacks. Each poisoner in the script can be run individually to generate a meta-database of itself. This can be done by running the `<poisoner>_generate_metadb.py` script in each poison repository.

# Installation

This guide outlines how to generate meta-databases for your experiments. Using a virtual environment (`venv`) is recommended to keep your environment organised and prevent dependency conflicts.


```bash
cd diva
python .\metadb_generation.py -n 10 -f results -s 0.05 -m 0.4
```

`-n:` Number of synthetic datasets to generate. 

`-f:` Output folder where results will be saved (default is Results/).

`-s:` Spacing between poisoning rates (default is 0.05).

`-m:` Maximum poisoning rate (default is 0.41).

This same command can be run to generate a test meta-database. 