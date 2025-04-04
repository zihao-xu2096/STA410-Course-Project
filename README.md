# STA410 Course Project: Bayesian Missing Data Imputation

This repository contains a Python package and demo notebook for Bayesian missing value imputation using MCMC methods such as Gibbs Sampling and Metropolis-Hastings. It is developed as part of the STA410 course project.

---

## Repository Structure

```
├── bayes_impute.py           # Main class: BayesianImputer
├── BayesDemo.ipynb           # Alternative or visual demo
├── README.md                 # This file
```

---

##  Project Objectives

- Support multiple missing data mechanisms (MCAR, MAR)
- Implement Bayesian multiple imputation using:
  - Gibbs Sampling
  - Metropolis-Hastings
- Generate multiple posterior-consistent imputations
- Provide visualization and diagnostics tools
- Compare with PyMC implementations
- Explore the impact of imputation on causal inference

> MNAR support is planned for future extension.

---

## Features

- Mean-based initialization for missing values
- Simple univariate normal models per column
- Gibbs and MH sampling with trace storage
- Automatic imputation of missing entries
- Easy switching between methods (`method='gibbs'` or `'metropolis'`)
- Supports custom number of imputations

---
##  References

- [PyMC missing data example](https://www.pymc.io/projects/examples/en/latest/howto/Missing_Data_Imputation.html)
- [Bayesian Data Analysis (Gelman et al.)](http://www.stat.columbia.edu/~gelman/book/)
- [Statistical Rethinking (McElreath)](https://xcelab.net/rm/statistical-rethinking/)

---

##  Authors

- Harry Zhao
- Shipeng Zhang
