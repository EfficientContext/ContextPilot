# ContextPilot

A Python package.

## Installation

```bash
pip install contextpilot
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ContextPilot.git
cd ContextPilot

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## Publishing to PyPI

### Prerequisites

1. Create accounts on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/)
2. Generate API tokens from your account settings

### Build and Publish

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to TestPyPI first (recommended)
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

### Using API Tokens

You can configure your API tokens in `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE

[testpypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
```

Or use environment variables:
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR-TOKEN-HERE
```

## License

Apache License 2.0