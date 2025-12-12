import numpy as np
from math import exp, pow, sin, cos, tan
import math

class Tensor:        
    def __init__(self, value, parents=None, _op=''):
        self.val = value
        self.parents = parents
        self.grad = 0
        self._op = _op


    def __add__(self, others):
        if isinstance(others, Tensor):
            out = Tensor(self.val + others.val, (self, others), '+') # we check if the other object is a instance Tensor class
        else: 
            out = Tensor(self.val + others, (self,), '+')

        def _backward(): # we implement the backward pass function here which calculates intermediate derivative
            self.grad += 1 * out.grad
            if isinstance(others, Tensor):
                others.grad += 1* out.grad

        out._backward = _backward
        return out
    
    def __radd__(self, const):
        return self.__add__(const) # when we add, const + Tensor 
    
    def __sub__(self, others):
        if isinstance(others, Tensor):
            out = Tensor(self.val - others.val, (self, others), '-')
        else:
            out = Tensor(self.val - others, (self,), '-') 

        def _backward():
            self.grad += 1 * out.grad
            if isinstance(others, Tensor):
                others.grad += -1* out.grad

        out._backward = _backward
        return out

    def __rsub__(self, others): 
        out = Tensor(others - self.val, (self,), '-')

        def _backward():
            self.grad += -1*out.grad
        out._backward = _backward
        return out

    
    def __mul__(self, others):
        if isinstance(others, Tensor):
            out = Tensor(self.val*others.val, (self, others), '*')
        else:
            out = Tensor(self.val * others, (self,), '*') 

        def _backward():
            if isinstance(others, Tensor):
                self.grad += others.val * out.grad
                others.grad += self.val * out.grad
            else:
                self.grad += others * out.grad

        out._backward = _backward

        return out
    
    def __rmul__(self, others):
        return self.__mul__(others)


    def __truediv__(self, others):
        if isinstance(others, Tensor):
            out = Tensor(self.val/others.val, (self,others), '/')
        else:
            out = Tensor(self.val/others, (self,), '/')
        
        def _backward():
            if isinstance(others, Tensor):
                self.grad += (1/others.val)*out.grad
                others.grad += -(self.val*others.val**-2)*out.grad
            else:
                self.grad += (1/others)*out.grad

        out._backward = _backward
        return out
    
    def __rtruediv__(self, others):
        out = Tensor(others/self.val, (self,), '/')

        def _backward():
            self.grad += -(others * self.val**-2)*out.grad
        out._backward = _backward

        return out
    
    @classmethod
    def e(cls, obj:Tensor):
        out = Tensor(exp(obj.val), (obj,), 'exp')
        def _backward():
            obj.grad += exp(obj.val)*out.grad
        out._backward = _backward
        return out
    
    @classmethod
    def power(cls, obj:Tensor, p:int|float):
        out = Tensor(pow(obj.val,p), (obj,), 'power')
        def _backward():
            obj.grad += p*pow(obj.val, p-1) * out.grad
        out._backward = _backward
        return out
    
    @classmethod 
    def sin_(cls, obj:Tensor):
        out = Tensor(sin(obj.val),(obj,), 'sin')
        def _backward():
            obj.grad += cos(obj.val) * out.grad
        out._backward = _backward
        return out
    
    @classmethod
    def cos_(cls, obj:Tensor):
        out = Tensor(cos(obj.val), (obj,), 'cos')
        def _backward():
            obj.grad += -sin(obj.val) * out.grad
        out._backward = _backward
        return out
    
    @classmethod
    def tan_(cls, obj:Tensor):
        out = Tensor(tan(obj.val), (obj,), 'tan')
        def _backward():
            obj.grad += 1/(cos(obj.val)**2) * out.grad
        out._backward = _backward
        return out
    
    @classmethod
    def relu(cls, obj):
        val = obj.val if obj.val > 0 else 0.0
        out = cls(val, (obj,), 'ReLU')

        def _backward():
            deriv = 1.0 if obj.val > 0 else 0.0
            obj.grad += deriv * out.grad
        out._backward = _backward
        return out
        
    @classmethod
    def sigmoid(cls, obj:Tensor):
        sig_val = 1.0 / (1.0 + math.exp(-obj.val))
        out = cls(sig_val, (obj,), 'Sigmoid')

        def _backward():
            deriv = sig_val * (1.0 - sig_val) 
            obj.grad += deriv * out.grad
        
        out._backward = _backward
        return out

    def invokeBackProp(self):
        self._backward()
        
    
def backward(func:Tensor):
    topo = []
    visited = set()

    def build_topo(v:Tensor):
        if v not in visited:
            visited.add(v)
            if v.parents == None: return
            for parent in v.parents:
                build_topo(parent)
            topo.append(v)

    build_topo(func)
    #seed gradient
    func.grad = 1.0

    for node in reversed(topo):
        node.invokeBackProp()

    
# Disclaimer: test cases are generated using AI
def test_autograd():
   # defining input variables
    W = Tensor(np.random.randn(1,10)*0.01)
    b = Tensor(np.zeros((1,1)))
    X = np.random.randn(10, 100)

    f = np.matmul(W, X) 
    print(f)
    print(f"Value of f: {f.val}")
    print("Gradients:")
    # initialize backward propagation
    backward(f)

    print(f"df_dw: {W.grad}")


# Run the test
if __name__ == "__main__":
    test_autograd()

