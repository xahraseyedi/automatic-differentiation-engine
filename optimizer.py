import numpy as np
from tensor import Tensor
from typing import List


# TODO
# Description: Implements the Stochastic Gradient Descent (SGD) optimizer.
# Optimizers are responsible for updating a model's parameters (weights and biases)
# based on the gradients computed during backpropagation. The goal is to minimize the
# loss function by iteratively adjusting the parameters in the opposite direction
# of their gradients.
#
# __init__:
# The constructor takes a list of the model's parameters (`params`) that it will
# manage and a learning rate (`lr`). The learning rate is a crucial hyperparameter
# that controls the size of the update steps.
class SGD:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("`params` must be a list of `Tensor` objects")

        self.params = params
        self.lr = lr

    # TODO
    # Description: Performs a single optimization step.
    # This method iterates through all the parameters registered with the optimizer and
    # updates their values. The update rule for standard SGD is:
    # `parameter.data = parameter.data - learning_rate * parameter.grad`
    #
    # Important: This operation should directly modify the `.data` attribute of each
    # parameter Tensor. It is an in-place update that should NOT be tracked by the
    # autograd engine (i.e., it should not create new nodes in the computation graph).
    def step(self):
        for param in self.params :
            param.data = param.data - self.lr * param.grad

    # TODO
    # Description: Resets the gradients of all registered parameters to zero.
    # Because gradients are accumulated (summed up) in the `.grad` attribute during
    # the backward pass, you must clear them before starting a new training iteration.
    # If you forget to do this, the gradients from the new batch will be added to the
    # gradients from the previous batch, leading to an incorrect optimization step.
    #
    # This method should be called at the beginning of each training loop, typically
    # right before the forward pass.
    def zero_grad(self):
        for param in self.params :
            param.grad = np.zeros_like( param.grad )

