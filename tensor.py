import numpy as np
from function import *
class Tensor:
    def __init__(self, data, requires_grad: bool = True, ctx=None):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self.ctx = ctx
    
    def __repr__(self):
        data_str = np.array2string(self.data, precision=4, suppress_small=True, prefix=' ' * 8)
        grad_str = np.array2string(self.grad, precision=4, suppress_small=True, prefix=' ' * 8)
        
        return (f"Tensor(shape={self.data.shape}\n"
                f" data: {data_str}\n"
                f" grad: {grad_str}")
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return Add.apply(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return self.__mul__(-1)
    
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __rsub__(self, other):
        return (-self).__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return Mul.apply(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return Pow.apply(self, other)
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return MatMul.apply(self, other)
    
    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis, keepdims)
    
    # TODO
    # Description: This method starts the entire backpropagation process.
    # It computes the gradient of the scalar output (typically a loss) with respect to
    # every other Tensor in the computation graph that has `requires_grad=True`.
    #
    # The process works in three main steps:
    #
    # 1. Initialization and Validation:
    #    - The backward pass must start from a scalar value (e.g., the final loss),
    #      as we are computing the gradient of this single value. We first check for this.
    #    - The gradient of the output Tensor with respect to itself (d(loss)/d(loss)) is 1.
    #      We initialize `self.grad` to 1 to kickstart the chain rule.
    #
    # 2. Topological Sort:
    #    - Backpropagation requires processing the computation graph in reverse order,
    #      from the output back to the inputs. A topological sort gives us an ordering of
    #      all Tensors (nodes) in the graph such that for every directed edge from
    #      node `u` to node `v`, `u` comes before `v` in the ordering.
    #    - The `build_topo` function performs this sort by recursively visiting each
    #      Tensor's parents (its `ctx.inputs`) and adding the Tensor to the `sorted_tensors`
    #      list only *after* all its parents have been visited.
    #
    # 3. The Backward Pass Loop:
    #    - We iterate through the `sorted_tensors` list in REVERSE order. This ensures
    #      that when we process a Tensor, the gradients for it (`t.grad`) have already
    #      been fully computed from all the subsequent operations that used it.
    #    - For each tensor `t` in the reversed list:
    #        a. We retrieve the function (`grad_fn`) and inputs from its context (`t.ctx`).
    #        b. We call the `backward` method of that function. This method takes the
    #           "upstream gradient" (`t.grad`) and computes the "local gradients"
    #           for its inputs.
    #        c. The computed gradients are then accumulated (`+=`) into the `.grad`
    #           attribute of each parent Tensor. We use `+=` because a single Tensor
    #           can be an input to multiple functions (e.g., y = w*x + w*z), and its
    #           total gradient is the sum of gradients from all paths.
    def backward(self):

        self.grad = np.ones_like(self.data, dtype=np.float32)


        visited = set()
        #بنویس
        topological_sort_tensors = []

        def sort ( tensor ):
            if tensor not in visited:
                visited.add(tensor)
                if tensor.ctx:
                    for i in tensor.ctx.inputs :
                        sort ( i )
                topological_sort_tensors.append( tensor )
        sort(self)
                
        for tensor in reversed ( topological_sort_tensors ) :
            if tensor.ctx is not None :
             #list of derivatives for each input variable
                grads = tensor.ctx.grad_fn.backward(tensor.ctx, tensor.grad)


             # update the gradient value of each input variable
                for i , grad in zip ( tensor.ctx.inputs , grads ):
                    if grad is not None:
                         i.grad = i.grad + grad
