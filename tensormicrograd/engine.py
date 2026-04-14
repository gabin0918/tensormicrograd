import numpy as np

class Tensor:
 

    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # leftover from micrograd, for graphviz / debugging / etc

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.data.shape}, grad={self.grad})"
    
    def _reduce_grad_dim(self, grad, original_shape):
        # Adjusting the shape of the gradient to match the shape of the original tensor
        
        # Gradient has to be the same shape as the original tensor, 
        # so we need to sum over the dimensions that were broadcasted
        while grad.ndim > len(original_shape):
            grad = grad.sum(axis=0) # e.g. [[1,2,3],[4,5,6]] -> [5,7,9]

        # If the original tensor had a dimension of size 1, 
        # we need to sum over that dimension as well
        for i, dim in enumerate(original_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True) # e.g. [[1,2],[3,4],[5,6]] -> [[3],[7],[11]]
        
        return grad

    def __add__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        #if tensors are differenrt shapes numPy will automatically broadcast them to the same shape
        out = Tensor(self.data + other.data,(self,other),'+')

        def _backward():
            self.grad += self._reduce_grad_dim(out.grad, self.data.shape)
            other.grad += self._reduce_grad_dim(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __mul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out=Tensor(self.data * other.data,(self,other),'*')  

        def _backward():
            self.grad += self._reduce_grad_dim(out.grad * other.data, self.data.shape)
            other.grad += self._reduce_grad_dim(out.grad * self.data, other.data.shape)
        out._backward = _backward

        return out
    
    def __matmul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,(self,other),'@')

        def _backward():
            
            self.grad += self._reduce_grad_dim(out.grad @ other.data.T, self.data.shape)
            other.grad += self._reduce_grad_dim(self.data.T @ out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, other):

        assert isinstance(other, (int, float)), "Supporting only int/float powers"
        
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():

            self.grad += (other * (self.data**(other-1))) * out.grad
            
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def sum(self):
        out = Tensor(np.sum(self.data), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        return self + (-other)

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __rsub__(self, other): # other - self
        return other + (-self)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # one differece from micrograd is that we need to set self.grad to an array of ones withhh data shape
        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()