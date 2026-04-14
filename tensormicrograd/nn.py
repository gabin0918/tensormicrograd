import numpy as np
from tensormicrograd.engine import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []
    
#Using Tensors, we vectorize 'Neuron' and 'Layer' into a single 'Linear' class for performance
class Linear(Module):
    def __init__(self, nin, nout, nonlin=True):
       
       # Initialize weights and biases with small random values
        super().__init__()
        self.w = Tensor(np.random.randn(nin, nout) * 0.1)
        self.b = Tensor(np.zeros((1, nout)))
        self.nonlin = nonlin

    def __call__(self, x):
        # Forward pass: y = x @ w + b, we use @ for performance
        act = x @ self.w + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"Linear(nin={self.w.data.shape[0]}, nout={self.w.data.shape[1]})"
    
class MLP(Module):
    """ A Multi-Layer Perceptron: a sequence of Linear layers """
    
    def __init__(self, nin, nouts):
        super().__init__()
        sz = [nin] + nouts
        #Create a list of Linear layers. 
        #The last layer is usually linear (nonlin=False) for regression/logits.
        self.layers = [
            Linear(sz[i], sz[i+1], nonlin=(i != len(nouts)-1)) 
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        #Pass the input through all layers in sequence
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        #Collect parameters from all layers recursively
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"