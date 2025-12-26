# RobustInfer Examples

This folder contains comprehensive examples demonstrating the functionality of both the Python and Scala libraries in the RobustInfer package.

## Python Library Examples (`python_examples/`)

The Python library provides JAX and PyTorch implementations of robust inference methods:

- **01_drgu_jax_basics.ipynb**: DRGU implementation using JAX - demonstrates the original implementation with automatic differentiation
- **02_drgu_pytorch_basics.ipynb**: DRGU implementation using PyTorch - compares JAX vs PyTorch implementations
- **03_drgu_minibatch.ipynb**: Mini-batch Fisher scoring for large datasets - demonstrates scalable training with momentum and learning rates
- **04_mwu_zero_trimmed_u.ipynb**: Non-parametric testing methods - Mann-Whitney U and Zero-trimmed U-statistics for zero-inflated positive data

## Scala Library Examples (`scala_examples/`)

The Scala library provides Spark-based distributed implementations optimized for big data:

- **01_drgu_basics.ipynb**: Basic DRGU implementation in Scala - demonstrates core functionality with Spark DataFrames
- **02_drgu_minibatch.ipynb**: Mini-batch DRGU with distributed processing - scalable implementation with anchor-based sampling
- **03_gee_examples.ipynb**: Generalized Estimating Equations (GEE) - various distribution families and correlation structures
- **04_twosample_tests.ipynb**: Two-sample testing methods - t-test, Mann-Whitney U, Zero-trimmed U comparisons
- **05_gee_ztu_simulation.ipynb**: Simulation studies on GEE and Zero-trimmed U methods

## Recent Updates

- **04_mwu_zero_trimmed_u.ipynb**: Fixed data generation to properly create zero-inflated positive data compatible with zero-trimmed U-statistics
- **03_gee_examples.ipynb**: Fixed Scala syntax errors for `case object` references (removed parentheses from `Gaussian()` and `Independent()`)

## Getting Started

### Prerequisites

**For Python examples:**
```bash
cd python_lib
pip install -e .
```

**For Scala examples:**
```bash
cd scala_lib
gradle build
```

### Running Examples

1. **Python examples**: Open in Jupyter Lab/Notebook with Python kernel
2. **Scala examples**: Use Almond Scala kernel or Spark-compatible notebook environment

### Key Concepts Demonstrated

- **DRGU (Doubly Robust Generalized U)**: Causal inference with U-statistics
- **Mini-batch optimization**: Scalable Fisher scoring for large datasets
- **GEE**: Generalized Estimating Equations for clustered data
- **Non-parametric tests**: Robust alternatives to parametric methods
- **Distributed computing**: Spark-based implementations for big data

## Performance Considerations

- **Python library**: Best for moderate-sized datasets (< 1M observations)
- **Scala library**: Designed for big data with Spark distributed processing
- **Mini-batch methods**: Enable training on datasets that don't fit in memory

## Statistical Background

Each notebook includes detailed explanations of the statistical methods, when to use them, and how to interpret results. The examples progress from basic usage to advanced scenarios including:

- Simulation studies comparing methods
- Performance benchmarking
- Convergence diagnostics
- Practical guidance for real-world applications
