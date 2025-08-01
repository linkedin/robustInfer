# robustInfer: Robust and Efficient Statistical Inference
RobustInfer provides implementation of modern robust statistical inference for online experimentation, e.g., regression adjustment, generalized estimating equations, Mann–Whitney U, zero-trimmed U, and Doubly Robust Generalized U. 

RobustInfer contains a Python library that is tailored for small to medium-scale analysis, as well as a Scala (Spark) library that is tailored for large-scale analysis. 

Theoretical details of the algorithms are described in the paper: https://arxiv.org/abs/2505.08128

## Copyright
Copyright 2025 LinkedIn Corporation
All Rights Reserved.

Licensed under the BSD 2-Clause License (the "License").
See [License](LICENSE) in the project root for license information.

## Usage
### Run Notebooks from Docker
1. Build the Docker Image:
```docker build -t robustinfer-notebook .```
2. Run the Docker Container:
```docker run -p 8888:8888 -v $(pwd):/app robustinfer-notebook```
The `-v $(pwd):/app` mounts the project directory into the container.
Access Jupyter Notebook at http://localhost:8888.
3. Example usage can be found in the notebooks folder.

### Build
- to build: ```make build```
- to clean: ```make clean```
- to run tests: ```make test```

## References
```
@article{wei2025beyond,
  title={Beyond Basic A/B testing: Improving Statistical Efficiency for Business Growth},
  author={Wei, Changshuai and Nguyen, Phuc and Zelditch, Benjamin and Chen, Joyce},
  journal={arXiv preprint arXiv:2505.08128},
  year={2025}
}

```



