# Simulation Studies

This folder contains simulation code used in the original research papers to validate the robust inference methods implemented in this library.

## Notebooks

- **`gee_vs_glm_simulation.ipynb`**: Compares GEE vs GLM for clustered/longitudinal data. Shows type I error control and power advantages of GEE.

- **`ugee_simulation.ipynb`**: Validates Unbiased Generalized Estimating Equations (UGEE) with U-statistics for causal inference.

- **`longitudinal_ugee_simulation.ipynb`**: Extends UGEE to longitudinal data with time-varying treatments and compound symmetric correlation.

- **`longitudinal_DRGU.ipynb`**: Implements Doubly Robust Generalized U-statistics (DRGU) for longitudinal treatment effects with model misspecification robustness.

- **`ZTU_mwu_ra_simulation.ipynb`**: Compares Zero-Trimmed U-statistics with Mann-Whitney U tests and regression adjustment. Demonstrates advantages with heavy-tailed distributions and excess zeros.

All simulations include type I error control, power analysis, and comparison with standard methods to validate the theoretical properties described in the original papers.
