# Inverse Reinforcement Learning: a New Framework to Mitigate an Intelligent Backoff Attack

## Introduction

Code used to obtain the results in the paper Parras, J., Almodóvar, A., Apellániz, P.A., & Zazo, S. (2022). Inverse Reinforcement Learning: a New Framework to Mitigate an Intelligent Backoff Attack. IEEE Internet of Things Journal, vol 9, no. 24, pp.24790-24799, December 2022. [DOI](https://doi.org/10.1109/JIOT.2022.3194694).

## Launch

To run this project, create a `virtualenv` (recomended) and then install the requirements as:

```
$ pip install -r requirements.txt
```
The results obtained in the paper are large (~4 GB), so they are not included in the repository. TO replicate the results, run the `main_imitation.py`file ensuring that you set the flag `train=True` and that you adjust the number of threads for your machine (take into account that the execution may take a while depending on your computer).  
