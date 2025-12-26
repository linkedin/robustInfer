# RobustInfer

A Python library for robust inference methods.

## Quick start

### Installation

#### Development Installation
For development with all dependencies including dev tools:

```bash
uv sync --locked --group dev
source .venv/bin/activate
```

Or using pip:
```bash
uv pip install -e . --group dev
```

#### Standard Installation
Install the basic package with PyTorch support:

```bash
pip install robustinfer
```

#### Installation with JAX Support
Install with optional JAX dependencies for additional DRGU implementation:

```bash
pip install robustinfer[jax]
```

#### Installation with All Optional Dependencies
Install with all optional dependencies:

```bash
pip install robustinfer[all]
```

#### Available Implementations
- **Default**: `DRGU` (PyTorch-based) - always available
- **Optional**: `DRGUJax` (JAX-based) - available when JAX is installed

### Usage

#### Basic Usage with PyTorch Implementation
```python
import robustinfer
import pandas as pd

# Load your data
data = pd.DataFrame({
    'treatment': [0, 1, 0, 1],
    'outcome': [1.2, 2.1, 1.5, 2.3],
    'covariate1': [0.5, 1.0, 0.8, 1.2],
    'covariate2': [2.1, 1.8, 2.0, 1.9]
})

# Create and fit DRGU model (PyTorch implementation)
model = robustinfer.DRGU(
    data=data,
    covariates=['covariate1', 'covariate2'],
    treatment='treatment',
    response='outcome'
)
model.fit()

# Get results
summary = model.summary()
print(summary)
```


## Development

### Running Tests
Run all tests, linting, and formatting

```bash
uv run tox run-parallel
```

Run tests for current version
```bash
uv run pytest -v
```

### Linting
Lint and fix code
```bash
uv run ruff check # Check only
uv run ruff check --fix # Check and fix
```

Format and update code
```bash
uv run ruff format
```
