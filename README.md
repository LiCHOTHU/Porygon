# Imitation

This repository is intended to be a jumping off point for imitation learning projects

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Follow these steps to set up the development environment:

### 1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository:
```bash
git clone https://github.com/yourusername/imitation.git
cd imitation
```

### 3. Create and sync a virtual environment:
```bash
uv sync
```

### 4. Install dependencies:
```bash
# Install main dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### 5. (optional) Install LIBERO
First download it
```bash
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd imitation
```
Next, since LIBERO is old we need to manually add the pyproject.toml
```bash
cp imitation/env/libero/pyproject.toml ../LIBERO/
```
Finally, install it
```bash
uv pip install ../LIBERO
```

### 6. (optional) Install MimicGen
First download it
```bash
cd ..
git clone https://github.com/NVlabs/mimicgen.git
cd imitation
```
Next, since MimicGen is old we need to manually add the pyproject.toml
```bash
cp imitation/env/mimicgen/pyproject.toml ../mimicgen/
```
Finally, install it
```bash
uv pip install ../mimicgen
```

### 7. (optional) Install DexMimicGen
This is a bit more involved since DexMimicGen runs on a different version of robosuite and there is no way to square the circle and make everything compatible. I would recommend setting this up in a separate conda environment.
First download it
```bash
cd ..
git clone https://github.com/NVlabs/dexmimicgen.git
cd imitation
```
Next go into the dexmimicgen repo and comment out all the lines in `requirements.txt`. __Do not skip this step!__

Now create a conda environment, activate it and install this package
```bash
conda create -n imitation-dmg python=3.10 -y
conda activate imitation-dmg
pip install -e .
```
I've found it works best if you install robosuite, robosuite_models and dexmimicgen from source.
```bash
cd ..
git clone -b v1.5.1 https://github.com/ARISE-Initiative/robosuite.git
pip instal robosuite
git clone https://github.com/ARISE-Initiative/robosuite_models.git
pip install robosuite_models
pip install -e ../dexmimicgen
```
Install robosuite_models
```bash
cd ..
cd imitation
```
Make sure you have the right version of mink
```bash
pip install mink==0.0.10
```

## Training

Example script to train a diffusion policy with a ResNet backbone on the LIBERO-90 benchmark

```bash
uv run train.py \
    --config-name=train.yaml \
    task=libero \
    algo=diffusion_policy \
    algo/encoder=rgb  \
    algo.chunk_size=8
```


## Development

The project uses several development tools:

- **Ruff** for linting and code formatting
- **MyPy** for static type checking
- **Pytest** for testing

To run the development tools:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy .

# Run tests
pytest
```

## Project Structure

```
imitation/
├── imitation/         # Main package
│   ├── algos/        # Algorithm implementations
│   ├── env/          # Environment code
│   └── utils/        # Utility functions
├── tests/            # Test files
├── config/           # Configuration files
├── scripts/          # Utility scripts
└── data/            # Data directory
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

