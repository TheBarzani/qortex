# Qortex
Modeling the financial markets using QML - Mil'Haq

## Installation and Setup

### Prerequisites
- Python 3.11
- pip or [uv](https://github.com/astral-sh/uv) package manager

### Method 1: Using pip (Standard)

1. **Clone the repository**
   ```bash
   git clone https://github.com/TheBarzani/qortex.git
   cd qortex
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   # .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Using uv (Fast and Modern)

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or
   pip install uv
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/TheBarzani/qortex.git
   cd qortex
   ```

3. **Create a virtual environment and install dependencies**
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate  # On macOS/Linux
   # or
   # .venv\Scripts\activate  # On Windows
   
   uv pip install -r requirements.txt
   ```

   Or in one command:
   ```bash
   uv pip sync requirements.txt
   ```

### Verify Installation

To verify your installation, start a Python interpreter and import the main packages:

```python
import numpy as np
import torch
import qiskit
import perceval
from merlinquantum import MerlinQuantum
print("All packages installed successfully!")
```

### Running Jupyter Notebooks

After installation, you can launch Jupyter to explore the notebooks:

```bash
jupyter notebook
```

Or using JupyterLab:

```bash
jupyter lab
```
