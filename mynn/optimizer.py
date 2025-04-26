from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]        # dangerous! should use unary operation! 


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        """
        Momentum gradient descent optimizer with self-contained velocity storage
        Args:
            init_lr: learning rate
            model: model containing layers to optimize
            mu: momentum coefficient (0.9 is typical)
        """
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocities = {}  # Dictionary to store velocities for each layer
    
    def step(self):
        """
        Perform a single optimization step using momentum
        For each parameter:
        1. Initialize velocity if not exists
        2. Apply weight decay if enabled
        3. Update velocity: v = μ*v - (1-μ)*grad
        4. Update parameters: θ = θ - lr*v
        """
        for layer in self.model.layers:
            if layer.optimizable == True:
                # Initialize velocities for this layer if not exists
                if layer not in self.velocities:
                    self.velocities[layer] = {
                        key: np.zeros_like(param)
                        for key, param in layer.params.items()
                    }
                
                for key in layer.params.keys():
                    # Apply weight decay if enabled
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    
                    # Update velocity terms (standard momentum formulation)
                    self.velocities[layer][key] = (
                        self.mu * self.velocities[layer][key]
                        - self.init_lr * layer.grads[key]
                    )
                    
                    # Update parameters using velocity
                    layer.params[key] += self.velocities[layer][key]