import os
import sys
import ctypes
import numpy as np

# 加入 LOMA 工程路径
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import compiler  # 引入 LOMA 编译器

# === NumPy 实现 ===
class KANNetwork:
    def __init__(self, layer_sizes, num_nonlinearities=6):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.num_nonlinearities = num_nonlinearities
        self.layers = []
        for i in range(self.num_layers):
            layer = KANLayer(layer_sizes[i], layer_sizes[i+1], num_nonlinearities)
            layer.initialize_weights()
            self.layers.append(layer)

    def forward(self, x):
        activations = x
        for layer in self.layers:
            s = np.dot(layer.weights, activations)
            outputs = np.zeros(layer.output_size)
            for i in range(layer.output_size):
                for q in range(layer.num_nonlinearities):
                    outputs[i] += layer.alpha_weights[i, q] * self.apply_nonlinearity(s[i], q)
            activations = outputs
        return activations

    def apply_nonlinearity(self, x, idx):
        if idx % 6 == 0:
            return 1.0 / (1.0 + np.exp(-x))
        elif idx % 6 == 1:
            return np.tanh(x)
        elif idx % 6 == 2:
            return np.maximum(0, x)
        elif idx % 6 == 3:
            return np.maximum(0.01 * x, x)
        elif idx % 6 == 4:
            return np.log(1.0 + np.exp(x))
        else:
            return x if x > 0 else np.exp(x) - 1

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

# === 主逻辑 ===
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kan_path = os.path.join(base_dir, "loma_code", "kan_exp2.py")
    
    with open(kan_path) as f:
        _, lib = compiler.compile(f.read(), target="c", output_filename="_code/kan_example")

    # 网络结构参数
    input_size = 2
    output_size = 2
    hidden_sizes = [3]
    num_nonlinearities = 6
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    # 初始化相同的权重
    np.random.seed(42)
    weights = []
    alphas = []
    for l in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[l]
        out_dim = layer_sizes[l+1]
        w = np.random.uniform(-0.1, 0.1, size=(out_dim, in_dim))
        a = np.random.uniform(0, 1, size=(out_dim, num_nonlinearities))
        a /= np.sum(a, axis=1, keepdims=True)
        weights.append(w)
        alphas.append(a)

    # NumPy 推理
    numpy_kan = KANNetwork(layer_sizes, num_nonlinearities)
    for l in range(len(weights)):
        numpy_kan.layers[l].weights = weights[l]
        numpy_kan.layers[l].alpha_weights = alphas[l]

    x_input = np.array([1.0, 2.0], dtype=np.float32)
    np_result = numpy_kan.forward(x_input)

    # 准备 ctypes 数据
    flat_weights = np.concatenate([w.flatten() for w in weights]).astype(np.float32)
    flat_alphas = np.concatenate([a.flatten() for a in alphas]).astype(np.float32)
    output_buffer = np.zeros(output_size, dtype=np.float32)

    # 转换为指针
    x_ptr = x_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    w_ptr = flat_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    a_ptr = flat_alphas.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_ptr = output_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # 设置函数签名
    lib.kan_network.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.kan_network.restype = None  # void

    # 执行函数
    lib.kan_network(x_ptr, w_ptr, a_ptr, y_ptr)

    # 输出对比
    print("NumPy KAN result:", np_result)
    print("LOMA KAN result:", output_buffer)