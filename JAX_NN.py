from tqdm import tqdm
from functools import partial
import time
import numpy as np
import jax.numpy as jnp
from jax import jit
import jax
import math
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

class Dense:
    def __init__(self, units:int, name:str, activation:str='linear'):
        self.units = units
        self.activation = activation
        self.name = name

class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=10e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
    def init_state(self, params):
        # Initialize optimizer state once params are available
        self.V = jax.tree.map(lambda x: jnp.zeros_like(x), params)
        self.S = jax.tree.map(lambda x: jnp.zeros_like(x), params)

    @partial(jit, static_argnums = 0)
    def param_update(self, params, grads, t):

        # Accumulation
        self.V = jax.tree.map(lambda v, g: v*self.beta1 + (1-self.beta1)*g, self.V, grads)
        self.S = jax.tree.map(lambda s, g: s*self.beta2 + (1-self.beta2)*g**2, self.S, grads)

        # Bias correction
        V_correct = jax.tree.map(lambda v: v/(1 - self.beta1**t),self.V)
        S_correct = jax.tree.map(lambda s: s/(1 - self.beta2**t), self.S)

        # Parameters update
        return jax.tree.map(lambda p, v, s: p - self.learning_rate*(v/(jnp.sqrt(s)+self.epsilon)), params, V_correct, S_correct )
        


class Dropout:
    pass

class NN:
    def __init__(self, input_dim:tuple[int], layers:tuple[Dense], dropout = None):
        self.input_dim = input_dim
        self.layers = layers
        self.dropout = dropout
        self.params = []

    def fit(self, X, Y, batch_size = 128, epochs = 100, lambda_=0):
        assert X.shape[1] == Y.shape[0]
        X, Y = jnp.array(X), jnp.array(Y)
        # Initializing random gaussian weights
        self.init_weights()

        # Initializing momentums(V, S)
        self.optimizer.init_state(self.params)

        num_iteration = math.floor(X.shape[1]/batch_size)
        for  j in range(1,epochs+1):
            loss = 0
            for i in tqdm(range(num_iteration), desc=f"epoch: {j}", ncols=70, ascii=False):

                mini_batch_X = X[:,i*batch_size:i*batch_size + batch_size]
                mini_batch_Y = Y[i*batch_size:i*batch_size + batch_size]
                # Backpropagation(includes forward propagation)
                gradients = self.backward(mini_batch_X, mini_batch_Y)
                # Update parameters after each backpropgation, grabs new gradients and update
                self.optimizer.t +=1
                self.params = self.optimizer.param_update(self.params, gradients, self.optimizer.t)
                # Forward prop to count loss

            loss = self.loss_accum(self.params, X, Y)
            print(f"loss:{float(loss):.4f}")

    
    def predict(self, X_test):
        return self.forward(self.params, X_test)

    def init_weights(self):
        current_input = self.input_dim[0]
        key = jax.random.key(42)
        for i, layer in enumerate(self.layers):
            key, subkey = jax.random.split(key)
            
            self.params.append(
                dict(W = jax.random.normal(subkey, shape=(layer.units, current_input)) * jnp.sqrt(2/current_input),
                    b = jnp.zeros((layer.units,1)))
                    )
            current_input = layer.units
    
    def forward(self, params, X):
        *hidden, last = params
        for layer in hidden:
            X = jax.nn.relu(layer['W'] @ X + layer['b'])
        return jax.nn.sigmoid(last['W'] @ X + last['b'])
    
    def _bce_loss(self, params,X,Y):
        a = self.forward(params, X)
        eps = 1e-8
        return -jnp.mean(Y*jnp.log(a+eps) + (1-Y)*jnp.log(1-a+eps))
    
    def backward(self, X, Y):
        grads = jax.grad(self.loss_accum,0)(self.params, X, Y)
        return grads
        
    def compile(self, optimizer, loss="bce"):
        if isinstance(optimizer, Adam):
            self.optimizer = optimizer
        if loss == "bce":
            self.loss_accum = self._bce_loss
        


if __name__ == "__main__":
    X, y = make_classification(n_samples = 1000000, n_features=50,n_classes=2, random_state=42)
    X = X.T
    l1 = Dense(32, "layer1", activation = "relu")
    l2 = Dense(32, "layer2", activation = "relu")
    l3 = Dense(1, "output", activation = "sigmoid")

    model = NN(X.shape,(l1,l2,l3))
    model.compile(
        optimizer = Adam(1e-4),
        loss = 'bce'
        )
    model.fit(X, y, batch_size = 128, epochs = 20)
    y_pred = model.predict(X) >= 0.5
    print(accuracy_score(y, y_pred))
