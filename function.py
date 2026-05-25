import numpy as np

__all__ = ["Add", "Mul", "Pow", "Log", "Sum", "MatMul", "ReLU", "Sigmoid", "Softmax", "CrossEntropyWithSoftmax"]

class Context:
    def __init__(self, grad_fn, *inputs):
        self.grad_fn = grad_fn
        self.inputs = inputs
            
class Function:
    @staticmethod
    def forward(ctx, *args): ...
    
    @staticmethod
    def backward(ctx, upstream_grad): ...

    @classmethod
    def apply(cls, *args):
        from tensor import Tensor
        inputs = [a for a in args if isinstance(a, Tensor)]
        ctx = Context(cls, *inputs)
        
        ctx.needs_input_grad = tuple(t.requires_grad for t in inputs)
        requires_grad = any(ctx.needs_input_grad)
        
        output_data = cls.forward(ctx, *args)
        ctx = ctx if requires_grad else None
        
        output_tensor = Tensor(output_data, requires_grad, ctx)
        
        return output_tensor

def unbroadcast_to(a, shape):
    while len(a.shape) > len(shape):
        a = a.sum(axis=0)
    
    for i, dim in enumerate(shape):
        if dim == 1:
            a = a.sum(axis=i, keepdims=True)

    return a

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data + b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = unbroadcast_to(upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        grad_b = unbroadcast_to(upstream_grad, ctx.b.data.shape) if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)
        
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data * b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = unbroadcast_to(ctx.b * upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        grad_b = unbroadcast_to(ctx.a * upstream_grad, ctx.b.data.shape) if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data @ b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = upstream_grad @ ctx.b.data.T if ctx.needs_input_grad[0] else None
        grad_b = ctx.a.data.T @ upstream_grad if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)
            
class Pow(Function):
    @staticmethod
    def forward(ctx, base, exp):
        ctx.base = base
        ctx.exp = exp
        ctx.output = np.pow(base.data, exp.data)
        return ctx.output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_base = ctx.exp.data * np.pow(ctx.base.data, ctx.exp.data - 1) * upstream_grad if ctx.needs_input_grad[0] else None
        grad_exp = np.log(ctx.base.data) * ctx.output * upstream_grad if ctx.needs_input_grad[1] else None
        
        return (grad_base, grad_exp)

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.a = a
        return np.log(a.data)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = 1/ctx.a.data * upstream_grad if ctx.needs_input_grad[0] else None
        return (grad_a,)
        
class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.a = a
        return np.sum(a.data, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = np.broadcast_to(upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        return (grad_a,)

# TODO
# Description: Implements the Rectified Linear Unit (ReLU) activation function.
# Forward pass: ReLU is defined as f(x) = max(0, x). It returns the input element
# if it is positive, and zero otherwise. You can use np.maximum for this.
# Backward pass: The derivative of ReLU is 1 for x > 0 and 0 for x <= 0.
# To implement this, you'll need to know which elements of the input were positive
# during the forward pass.
# Hint: Save a boolean mask (e.g., a.data > 0) in the `ctx` during the forward pass.
# Use this mask in the backward pass to multiply with the upstream_grad, effectively
# zeroing out the gradient for elements that were not positive.
class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.a = a
        ctx.mask = a.data > 0
        return np.maximum( 0 , a.data )
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = upstream_grad * ctx.mask if ctx.needs_input_grad[0] else None
        return (grad_a,)
        

# TODO
# Description: Implements the Sigmoid activation function.
# Forward pass: Sigmoid is defined as f(x) = 1 / (1 + exp(-x)).
# Backward pass: The derivative of the sigmoid function with respect to its input x is
# f'(x) = f(x) * (1 - f(x)).
# Hint: Instead of re-calculating the sigmoid in the backward pass, a common and
# efficient trick is to save the *output* of the forward pass in the `ctx`.
# You can then use this saved output to easily compute the derivative.
class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.a = a
        output = 1/(1+np.exp(-1*a.data))
        ctx.y = output
        return output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = ctx.y*(1-ctx.y)*upstream_grad
        return (grad_a,)

# TODO
# Description: Implements the Softmax activation function.
# Forward pass: Softmax converts a vector of numbers (logits) into a probability distribution.
# The formula is S(x_i) = exp(x_i) / sum(exp(x_j)) for all j.
# A crucial part of implementing softmax is ensuring numerical stability. If the inputs `a`
# are large, `np.exp(a.data)` can overflow.
# Hint: Use the "max-subtraction trick": subtract the maximum value of the input along the
# specified axis from all elements before exponentiating. This doesn't change the output but
# prevents overflow. `a_stable = a.data - a.data.max(axis=axis, keepdims=True)`.
#
# Backward pass: The derivative of softmax is more complex than other activations because
# the output of one element depends on all other elements.
# The local gradient is `output * (upstream_grad - np.sum(upstream_grad * output, axis=..., keepdims=True))`
# Hint: Save the output of the forward pass in the `ctx` to reuse it here.
class Softmax(Function):
    @staticmethod
    def forward(ctx, a, axis):
        ctx.a = a
        #we use a_prim instead of 'a' to avoid overflow
        a_prim = a.data - a.data.max( axis = axis , keepdims = True )
        output =  np.exp( a_prim ) / np.sum( np.exp( a_prim ), axis = axis , keepdims = True )
        ctx.output = output
        ctx.axis = axis
        return output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        S = ctx.output
        dot = np.sum(upstream_grad * S, axis=ctx.axis, keepdims=True)
        grad_input = S * (upstream_grad - dot)
        return (grad_input,)
        
            
# TODO
# Description: Implements the Cross-Entropy Loss combined with a Softmax activation.
# This is a common pattern for multi-class classification problems. Combining them into
# a single function provides a simpler and more numerically stable gradient.
#
# Forward pass:
# 1. Apply the numerically stable softmax function to the predictions `y_pred`.
# 2. Compute the cross-entropy loss: L = -sum(y_true * log(softmax_output)).
# 3. Normalize the loss by dividing the sum by the number of samples (batch size).
# The batch size is typically the size of the first dimension of `y_pred`.
#
# Backward pass:
# The magic of combining these two is the resulting gradient. The gradient of the
# Cross-Entropy Loss with respect to the *pre-softmax* inputs (`y_pred`) is simply:
# `(softmax_output - y_true)`.
# Remember to also normalize this gradient by the batch size, just as you did for the loss.
#
# Hint: In the forward pass, you'll need to save `softmax_output` and `y_true` in the `ctx`
# to use them in the backward pass. The `upstream_grad` for a final loss function is typically 1.0.
class CrossEntropyWithSoftmax(Function):
    @staticmethod
    def forward(ctx, y_pred, y_true):
        
        ctx.y_true = y_true.data
        shifted_logits = y_pred.data - np.max(y_pred.data, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        ctx.probs = probs

        batch_size = y_pred.data.shape[0]
        correct_logprobs = -np.log(probs[np.arange(batch_size), np.argmax(ctx.y_true, axis=1)])
        loss = np.sum(correct_logprobs) / batch_size

        return loss
       
    
    @staticmethod
    def backward(ctx, upstream_grad):
        
        batch_size = ctx.probs.shape[0]
        grad = ctx.probs.copy()
        grad[np.arange(batch_size), np.argmax(ctx.y_true, axis=1)] -= 1
        grad /= batch_size
        
        return (upstream_grad * grad, None)
