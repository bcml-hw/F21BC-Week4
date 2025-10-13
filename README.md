# F21BC - Biologically Inspired Computation - Week 4 Lab Materials

Hello everyone. This repository contains the lab materials for Week 4 of the F21BC course. All code has been tested on Google Colab (September 2025). If you encounter issues in the future, please check the versions of the packages you are using.

## Repository Structure

### `/data`
- Contains the data files required for running the example code in this lab

### `/keras&tensorflow`
- Example code using Keras and TensorFlow
- Includes solutions for all four tasks as a reference

### `/pytorch` 
- PyTorch versions of the examples
- **Does NOT include solutions** - Interested students could consult relevant documentation or use AI assistance to implement their own solutions

### `requirements.txt`
- Packages listed from the Google Colab environment
- **Not all packages are strictly required** for running the example code. Here is a reference for package versions if you encounter compatibility issues

### `Install_Conda_To_Colab_And_Install_Tensorflow1.x.ipynb`
- Some lab in other weeks may require TensorFlow 1.x, which is no longer supported in the current Google Colab environments. This notebook provides a workaround by installing Miniconda in Google Colab and setting up a Python 3.6 environment with TensorFlow 1.15 (seems the only Python version for supporting tensorflow1.x at present)
- **Note**: This script is designed for Linux systems or Google Colab. Some commands may differ between Windows and Linux systems. If you want to use a Windows Python environment with your own local machine, you should consult the appropriate installation and activation methods for Miniconda

## Important Notes

⚠️ **Compatibility Warning**: The code was tested in September 2025 with Google Colab. Package updates may cause compatibility issues in the future. Check package versions if you encounter errors.

🔧 **TensorFlow 1.x Users**: For labs requiring TensorFlow 1.x, use the provided Conda installation script as current Google Colab environments no longer support TensorFlow 1.x directly.

🎯 **Learning Focus**: 
- The provided solutions in `/keras&tensorflow` are **reference implementations**, not the only correct way to solve the problems.
- We strongly encourage you to understand the concepts and **implement your own solutions**. Experimenting and developing your own approach is a valuable part of the learning process.
