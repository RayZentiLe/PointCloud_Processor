#!/bin/bash
set -e

# Usage: ./setup_env.sh [env_name]
# If env_name is provided, use it. Otherwise create "pcprocessor".

ENV_NAME="${1:-pcprocessor}"

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH."
    echo "Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

eval "$(conda shell.bash hook)"

# Check if environment already exists
if conda env list | grep -qw "^${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists. Activating..."
    conda activate "${ENV_NAME}"
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.11..."
    conda create -n "${ENV_NAME}" python=3.11 -y
    conda activate "${ENV_NAME}"
fi

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install numpy scipy
pip install PySide6
pip install vtk
pip install open3d
pip install PyOpenGL PyOpenGL_accelerate

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Environment: ${ENV_NAME}"
echo ""
echo "  To run the application:"
echo "    conda activate ${ENV_NAME}"
echo "    cd src && python main.py"
echo ""
echo "  Or use:  ./run.sh ${ENV_NAME}"
echo "============================================"