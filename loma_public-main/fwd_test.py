import ctypes
import numpy as np
import torch
import compiler
import parser
import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
from kan_modules import kan_forward_diff

# 编译环境初始化
structs, funcs, diff_structs = {}, {}, {}

# 用户定义的主函数
def_code = """
def kan_network(X: In[Array[float, 2]], W: In[Array[float, 12]], A: In[Array[float, 30]], Y: Out[Array[float, 2]]):
    kan(X, 2, 2, [3], 6, W, A, Y)

d_kan_network = fwd_diff(kan_network)
"""

funcs.update(parser.parse(def_code)[1])

kan_forward_diff.create_param_kan_forward_diff(
    diff_func_id="d_kan_network",
    structs=structs,
    funcs=funcs,
    diff_structs=diff_structs,
    input_size=2,
    output_size=2,
    hidden_sizes=[3],
    nonlinearities=["sigmoid", "tanh", "relu", "leaky_relu", "softplus", "elu"]
)

ctypes_structs, lib = compiler.compile(def_code, target="c", output_filename="_code/kan_example")
DFloat = ctypes_structs["_dfloat"]

input_size = 2
output_size = 2
x = np.array([1.0, 2.0], dtype=np.float32)
dx = np.array([1.0, 0.0], dtype=np.float32)

W_np = np.array([
    -0.7442485,  -0.67011464, -0.38753605,  0.8150472,
     0.2024877,  -1.1855903,   0.8120937,   1.5146936,
    -0.2709462, -0.220293,    -0.09243973, -1.8649737
], dtype=np.float32)

A_np = np.array([
     0.4986072,   0.32935795, -0.5030321,  -0.7154783,   1.5632602,   0.01966095,
    -0.2189199,  -0.02422015, -0.8357302,  -1.1509243,   1.099332,   -0.60079235,
    -1.1813881,   0.34485862, -0.7276707,   1.4125842,  -0.54942584,  0.5159596,
    -0.607898,   -1.0768197,  -1.0515252,   0.37554744, -0.8326384,  -0.84356564,
     0.38193658, -0.3604392,  -0.91708106, -1.1707342,  -1.755714,   -0.35679936
], dtype=np.float32)

W = (DFloat * len(W_np))(*[DFloat(w, 0.0) for w in W_np])
A = (DFloat * len(A_np))(*[DFloat(a, 0.0) for a in A_np])

X = (DFloat * input_size)(*[
    DFloat(val, dval) for val, dval in zip(x, dx)
])
Y = (DFloat * output_size)()

lib.d_kan_network(X, W, A, Y)
print(W_np)
print(A_np)
print("C-based ForwardDiff Output:")
for i in range(output_size):
    print(f"Y[{i}].val = {Y[i].val:.6f}, dval = {Y[i].dval:.6f}")

# 手动 NumPy 实现以进行前向与导数调试
class KANLayer:
    def __init__(self, input_size, output_size, num_nonlinearities):
        self.input_size = input_size
        self.output_size = output_size
        self.num_nonlinearities = num_nonlinearities
        self.weights = None
        self.alpha_weights = None

    def initialize_weights(self):
        self.weights = np.random.randn(self.output_size, self.input_size) * 0.1
        self.alpha_weights = np.random.rand(self.output_size, self.num_nonlinearities)
        self.alpha_weights /= np.sum(self.alpha_weights, axis=1, keepdims=True)

class KANNetwork:
    def __init__(self, layer_sizes, weights, alphas):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.num_nonlinearities = 6
        self.weights = weights
        self.alphas = alphas

    def forward(self, x):
        activations = x
        idx_w = 0
        idx_a = 0
        for l in range(self.num_layers):
            in_size = self.layer_sizes[l]
            out_size = self.layer_sizes[l+1]
            W = self.weights[idx_w:idx_w + in_size * out_size].reshape(out_size, in_size)
            A = self.alphas[idx_a:idx_a + out_size * 6].reshape(out_size, 6)
            idx_w += in_size * out_size
            idx_a += out_size * 6
            s = np.dot(W, activations)
            outputs = np.zeros(out_size)
            for i in range(out_size):
                for q in range(6):
                    outputs[i] += A[i, q] * self.apply_nonlinearity(s[i], q)
            activations = outputs
        return activations

    def apply_nonlinearity(self, x, idx):
        if idx == 0:
            return 1.0 / (1.0 + np.exp(-x))
        elif idx == 1:
            return np.tanh(x)
        elif idx == 2:
            return np.maximum(0, x)
        elif idx == 3:
            return np.maximum(0.01 * x, x)
        elif idx == 4:
            return np.log(1.0 + np.exp(x))
        else:
            return x if x > 0 else np.exp(x) - 1

np_kan = KANNetwork([2, 3, 2], W_np, A_np)
print("\nNumPy KAN Forward Only:", np_kan.forward(x))

# C-based primal-only output using raw float call
x_input = x.astype(np.float32)
weights = [W_np[:6].reshape(3, 2), W_np[6:].reshape(2, 3)]
alphas = [A_np[:18].reshape(3, 6), A_np[18:].reshape(2, 6)]
flat_weights = np.concatenate([w.flatten() for w in weights]).astype(np.float32)
flat_alphas = np.concatenate([a.flatten() for a in alphas]).astype(np.float32)
output_buffer = np.zeros(output_size, dtype=np.float32)

x_ptr = x_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
w_ptr = flat_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
a_ptr = flat_alphas.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
y_ptr = output_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

lib.kan_network.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]
lib.kan_network.restype = None

lib.kan_network(x_ptr, w_ptr, a_ptr, y_ptr)
print("\nLOMA KAN result:", output_buffer)
