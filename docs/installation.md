# Installation

## Requirements

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment

## Quick Install

### 1. Clone the Repository

```bash
git clone https://github.com/jayashankarvr/priorpatch.git
cd priorpatch
```

### 2. Create Virtual Environment (Recommended)

**Using venv:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**

```bash
conda create -n priorpatch python=3.10
conda activate priorpatch
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PriorPatch

**Development mode (editable install):**

```bash
pip install -e .
```

**Regular install:**

```bash
pip install .
```

## Verify Installation

Check that PriorPatch is installed correctly:

```bash
priorpatch --help
```

You should see the help message with available commands.

**Test with sample image:**

```bash
priorpatch analyze --input examples/sample_input.png --outdir test_output/
```

## Dependencies

PriorPatch requires the following packages:

### Core Dependencies

- **numpy** (>=1.21.0): Numerical computing
- **Pillow** (>=9.0.0): Image loading/saving
- **matplotlib** (>=3.5.0): Visualization and heatmap generation
- **scipy** (>=1.7.0): Signal processing (FFT, DCT, filters)

### Documentation (Optional)

- **mkdocs** (>=1.4.0): Documentation site generator
- **mkdocs-material** (>=9.0.0): Material theme for MkDocs

### Testing (Optional)

- **pytest** (>=7.0.0): Testing framework
- **pytest-cov** (>=4.0.0): Coverage reporting

## Installation Methods

### Method 1: Development Install (Recommended for Contributors)

This method creates a symlink, so code changes take effect immediately:

```bash
git clone https://github.com/jayashankarvr/priorpatch.git
cd priorpatch
pip install -e .[dev]
```

### Method 2: User Install

For end users who just want to use the tool:

```bash
git clone https://github.com/jayashankarvr/priorpatch.git
cd priorpatch
pip install .
```

### Method 3: Install from Source Tarball

```bash
# Download source tarball
wget https://github.com/jayashankarvr/priorpatch/archive/v0.1.0.tar.gz
tar -xzf v0.1.0.tar.gz
cd priorpatch-0.1.0
pip install .
```

## Optional Components

### Install with Development Tools

```bash
pip install -e .[dev]
```

This includes pytest, coverage, and documentation tools.

### Install Documentation Tools Only

```bash
pip install mkdocs mkdocs-material
```

Then build and serve documentation:

```bash
mkdocs serve
```

Visit <http://localhost:8000> to view docs.

## Platform-Specific Notes

### Linux

No special requirements. Standard installation should work.

### macOS

If you encounter issues with scipy:

```bash
# Install Xcode command line tools
xcode-select --install

# Install via Homebrew (alternative)
brew install python
pip install -r requirements.txt
```

### Windows

**Option 1: Use Anaconda/Miniconda (Recommended)**

Anaconda comes with pre-compiled scientific packages:

```bash
conda install numpy scipy matplotlib pillow
pip install .
```

**Option 2: Use Windows Subsystem for Linux (WSL)**

Install WSL and follow Linux instructions.

**Option 3: Native Windows Install**

Ensure Visual C++ Build Tools are installed if you encounter compilation errors:

- Download from: <https://visualstudio.microsoft.com/visual-cpp-build-tools/>

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'priorpatch'`

**Solution**:

```bash
# Ensure you're in the correct environment
pip install -e .
```

### scipy Installation Issues

**Problem**: scipy fails to install or compile

**Solution**: Use conda or pre-built wheels

```bash
conda install scipy
# OR
pip install --only-binary :all: scipy
```

### Permission Errors

**Problem**: Permission denied during installation

**Solution**: Use virtual environment or `--user` flag

```bash
pip install --user -e .
```

### Version Conflicts

**Problem**: Dependency version conflicts

**Solution**: Use fresh virtual environment

```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Updating PriorPatch

### From Git Repository

```bash
cd priorpatch
git pull origin main
pip install -e . --upgrade
```

### Check Current Version

```python
import priorpatch
print(priorpatch.__version__)
```

Or from command line:

```bash
pip show priorpatch
```

## Uninstalling

```bash
pip uninstall priorpatch
```

To remove all dependencies as well:

```bash
pip freeze | grep -v "^-e" | xargs pip uninstall -y
```

## Docker Installation (Advanced)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["priorpatch", "--help"]
```

Build and run:

```bash
docker build -t priorpatch .
docker run -v $(pwd)/data:/data priorpatch analyze --input /data/image.jpg --outdir /data/output
```

## Next Steps

After installation:

1. Read the [Usage Guide](usage.md)
2. Try the example in `examples/`
3. Explore the [API Reference](api.md)
4. Check out [Detector Details](detectors.md)
