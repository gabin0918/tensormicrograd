import torch
import numpy as np
from tensormicrograd.engine import Tensor

def test_broadcasting_complex():
    """Tests automatic gradient reduction for different shapes (Broadcasting)"""
    # A (3, 2) + B (1, 2) -> Result (3, 2)
    a_data = np.random.randn(3, 2)
    b_data = np.random.randn(1, 2)
    
    a = Tensor(a_data)
    b = Tensor(b_data)
    out = (a + b) * a # Element-wise addition + multiplication
    loss = out.sum()
    loss.backward()
    
    # PyTorch
    a_pt = torch.tensor(a_data, requires_grad=True)
    b_pt = torch.tensor(b_data, requires_grad=True)
    out_pt = (a_pt + b_pt) * a_pt
    loss_pt = out_pt.sum()
    loss_pt.backward()
    
    assert np.allclose(loss.data, loss_pt.data.item())
    assert np.allclose(a.grad, a_pt.grad.numpy())
    assert np.allclose(b.grad, b_pt.grad.numpy())
    print("Test broadcasting_complex passed! (Gradient reduction works)")

def test_deep_linear_chain():
    """Tests a chain of matrix operations (similar to a real neural network)"""
    X_data = np.random.randn(4, 3) # Batch 4, Features 3
    W1_data = np.random.randn(3, 5)
    b1_data = np.random.randn(1, 5)
    W2_data = np.random.randn(5, 1)
    
    # Your engine
    X = Tensor(X_data)
    W1 = Tensor(W1_data)
    b1 = Tensor(b1_data)
    W2 = Tensor(W2_data)
    
    # Graph: ReLU(X @ W1 + b1) @ W2
    h = (X @ W1 + b1).relu()
    y = h @ W2
    loss = y.sum()
    loss.backward()
    
    # PyTorch
    X_pt = torch.tensor(X_data, requires_grad=True)
    W1_pt = torch.tensor(W1_data, requires_grad=True)
    b1_pt = torch.tensor(b1_data, requires_grad=True)
    W2_pt = torch.tensor(W2_data, requires_grad=True)
    
    h_pt = torch.relu(X_pt @ W1_pt + b1_pt)
    y_pt = h_pt @ W2_pt
    loss_pt = y_pt.sum()
    loss_pt.backward()
    
    assert np.allclose(W1.grad, W1_pt.grad.numpy())
    assert np.allclose(b1.grad, b1_pt.grad.numpy())
    assert np.allclose(W2.grad, W2_pt.grad.numpy())
    print("Test deep_linear_chain passed! (MatMul + ReLU + Add works)")

def test_tensor_reuse_branching():
    """Tests if gradients accumulate correctly when a single tensor is used in multiple branches"""
    x_data = np.random.randn(2, 2)
    
    x = Tensor(x_data)
    # y = x^2 + x.sum()
    # Here x affects the result in two different ways
    y = (x ** 2) + x.sum() 
    z = y.sum()
    z.backward()
    
    x_pt = torch.tensor(x_data, requires_grad=True)
    y_pt = (x_pt ** 2) + x_pt.sum()
    z_pt = y_pt.sum()
    z_pt.backward()
    
    assert np.allclose(x.grad, x_pt.grad.numpy())
    print("Test tensor_reuse_branching passed! (Gradient accumulation works)")

if __name__ == "__main__":
    test_broadcasting_complex()
    test_deep_linear_chain()
    test_tensor_reuse_branching()