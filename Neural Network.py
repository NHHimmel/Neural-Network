import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import jax.random as random
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
np.random.seed(42)

class RMSProp:
    def __init__(self, params, learning_rate=0.001, beta=0.9, eps=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.eps = eps
        self.G = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]

    def gradient_update(self, params, gradients):
        new_params = []
        for i, ((W, b), (dW, db)) in enumerate(zip(params, gradients)):
            gw, gb = self.G[i]
            # Exponential moving average of squared gradients
            gw = self.beta*gw + (1-self.beta)*(dW ** 2)
            gb = self.beta*gb + (1-self.beta)*(db ** 2)

            # Scale gradients
            W_new = W-self.lr*dW/(np.sqrt(gw) + self.eps)
            b_new = b-self.lr*db/(np.sqrt(gb) + self.eps)

            self.G[i] = (gw, gb)
            new_params.append((W_new, b_new))
        return new_params


class Layer:
    def __init__(self, name:str, units:int, activation:str="linear")->None:
        self.units = units
        self.activation = activation
        self.name = name

class NN:
    def __init__(self, input_dim:tuple, n_layers:tuple[Layer])->None:
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.params = []

    def fit(self, X:jax.Array, Y:jax.Array, epochs:int=100, learning_rate:float=0.01)->None:
        assert X.shape[1] == Y.shape[0]
        self._init_weights()
        rms_prop = RMSProp(self.params, learning_rate)
        for i in range(1,epochs+1):
            gradients = self._backward_pass(X, Y)
            self.params = rms_prop.gradient_update(self.params, gradients)
            loss = self._loss_func(self.params, X, Y)
            if i % 100 == 0:
                print(f"epoch:{i} ---- loss:{loss:.4f}")

    def predict_proba(self, X:jax.Array)->jax.Array[float]:
        return self._forward_pass(self.params, X)

    def predict(self, X:jax.Array)->jax.Array[int]:
        A = self._forward_pass(self.params, X)
        return jnp.where(A >= 0.5, 1, 0)

    def _init_weights(self)->None:
        current_input = self.input_dim[0]
        self.params = []
        for i, layer in enumerate(self.n_layers):
            W = jnp.array(np.random.randn(layer.units, current_input))*0.01
            b = jnp.zeros((layer.units, 1)) 
            self.params.append((W, b))
            current_input = layer.units

    def _forward_pass(self, params:list[tuple[jax.Array]], X:jax.Array)->jax.Array:
        A = X
        for i, layer in enumerate(self.n_layers):
            W, b = params[i]
            Z = jnp.dot(W, A) + b
            if layer.activation == "sigmoid":
                A = self._sigmoid(Z)
            elif layer.activation == "relu":
                A = self._relu(Z)
        return A

    def _loss_func(self, params:list[tuple[jax.Array]], X:jax.Array, Y:jax.Array):
        A_l = self._forward_pass(params, X)
        eps = 1e-7
        return -jnp.mean(Y*jnp.log(A_l+eps) + (1-Y)*jnp.log(1-A_l+eps))

    def _backward_pass(self, X:jax.Array, Y:jax.Array):
        grad_func = jax.grad(lambda params: self._loss_func(params, X, Y))
        return grad_func(self.params) 

    def _params_update(self, params:list[tuple[jax.Array]], gradients, learning_rate:float):
        new_params = []
        for (W, b), (dW, db) in zip(params, gradients):
            new_params.append((W - learning_rate*dW, b - learning_rate*db))
        return new_params

    @staticmethod
    def _sigmoid(Z):
        return 1/(1+jnp.exp(-Z))

    @staticmethod
    def _relu(Z):
        return jnp.where(Z > 0, Z, 0)

if __name__ == "__main__":
    X, y = make_classification(n_samples = 100000, n_features=30,n_classes=2, random_state=42)
    X = X.T
    l1 = Layer(name="layer1", units=10, activation="relu")
    l2 = Layer(name="layer2", units=10, activation="relu")
    l3 = Layer(name="output", units=1, activation="sigmoid")

    model = NN(X.shape, (l1, l2, l3))
    model.fit(X, y, epochs=5000, learning_rate=0.005)
    y_pred = model.predict(X)
    print(accuracy_score(y, y_pred.T))
