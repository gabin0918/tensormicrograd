# TensorMicrograd

A lightweight, tensor-based autograd engine written in Python and NumPy. Inspired by Andrej Karpathy's `micrograd`, this project extends the scalar-based approach to support N-dimensional arrays (Tensors).

## Key Features
- **Vectorized Operations**: Uses NumPy for fast matrix multiplications (`@`) and element-wise operations.
- **Broadcasting Support**: Automatically handles gradient reduction for tensors with different shapes (just like PyTorch).
- **Neural Network Library**: High-level `Linear` and `MLP` classes for easy building of deep learning models.

## Installation

```bash
git clone https://github.com/gabin0918/tensormicrograd.git
cd tensormicrograd
pip install -e .
## Example Usage

```python
from tensormicrograd.engine import Tensor
from tensormicrograd.nn import MLP

# Create a Multi-Layer Perceptron (2 inputs, hidden layers [16, 16], 1 output)
model = MLP(2, [16, 16, 1])

# Forward pass with a batch of data
x = Tensor([[2.0, 3.0], [1.0, -1.0]])
y = model(x)

# Backward pass to compute gradients
y.sum().backward() 

print(f"Input gradients:\n{x.grad}")

## Running Tests
This engine is verified against PyTorch to ensure mathematical correctness of gradients and broadcasting logic.

```bash
python -m test.test_eng