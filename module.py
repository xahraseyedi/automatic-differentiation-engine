import numpy as np
from tensor import Tensor
from function import ReLU as ReLU_Func

class Module:
    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def set_parameters(self, params):
        param_list = self.parameters()
        for i, p_data in enumerate(params):
            param_list[i].data = p_data
    
    def forward(self, *args, **kwargs): ...
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        weight_data = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.weight = Tensor(weight_data, requires_grad=True)
        
        bias_data = np.zeros(out_features)
        self.bias = Tensor(bias_data, requires_grad=True)
    def forward(self, x):
        return x @ self.weight + self.bias

class ReLU(Module):
    def forward(self, x):
       return ReLU_Func.apply(x)

