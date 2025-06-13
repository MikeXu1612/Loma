import ctypes
import numpy as np
import timeit
import torch
import torch.nn.functional as F
import torch.nn as nn

# ======= 编译并加载 C 函数 =======
import compiler
import parser
import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
from kan_modules import kan_forward_diff

def_code = """
def kan_network(X: In[Array[float, 2]], W: In[Array[float, 12]], A: In[Array[float, 30]], Y: Out[Array[float, 2]]):
    kan(X, 2, 2, [3], 6, W, A, Y)

d_kan_network = fwd_diff(kan_network)
"""

structs, funcs, diff_structs = {}, {}, {}
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

# ======= 数据准备 =======
x = np.array([1.0, 2.0], dtype=np.float32)
dx = np.array([1.0, 0.0], dtype=np.float32)
output_size = 2

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

# ======= PyTorch 实现 =======
class TorchKAN(nn.Module):
    def __init__(self, weights, alphas):
        super().__init__()
        self.weights = [torch.tensor(w, dtype=torch.float32) for w in weights]
        self.alphas = [torch.tensor(a, dtype=torch.float32) for a in alphas]

    def forward(self, x):
        for w, a in zip(self.weights, self.alphas):
            s = F.linear(x, w)
            out = torch.zeros_like(s)
            for i in range(s.shape[0]):
                s_i = s[i]
                nonlinear = torch.stack([
                    torch.sigmoid(s_i),
                    torch.tanh(s_i),
                    F.relu(s_i),
                    F.leaky_relu(s_i, 0.01),
                    torch.log1p(torch.exp(s_i)),
                    s_i if s_i > 0 else torch.exp(s_i) - 1,
                ])
                out[i] = torch.dot(a[i], nonlinear)
            x = out
        return x

# 解析结构
weights = [W_np[:6].reshape(3, 2), W_np[6:].reshape(2, 3)]
alphas = [A_np[:18].reshape(3, 6), A_np[18:].reshape(2, 6)]

torch_kan = TorchKAN(weights, alphas)

# ======= C 调用封装 =======
X = (DFloat * 2)(*[DFloat(v, d) for v, d in zip(x, dx)])
W = (DFloat * len(W_np))(*[DFloat(v, 0.0) for v in W_np])
A = (DFloat * len(A_np))(*[DFloat(v, 0.0) for v in A_np])
Y = (DFloat * 2)()

# 原始值版本（只使用 val）
x_val = x.astype(np.float32)
w_flat = np.concatenate([w.flatten() for w in weights]).astype(np.float32)
a_flat = np.concatenate([a.flatten() for a in alphas]).astype(np.float32)
y_buf = np.zeros(2, dtype=np.float32)

x_ptr = x_val.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
w_ptr = w_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
a_ptr = a_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
y_ptr = y_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

lib.kan_network.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4
lib.kan_network.restype = None

# ======= Benchmark =======
n_runs = 1

torch_time = timeit.timeit(lambda: torch_kan(torch.tensor(x)).detach().numpy(), number=n_runs)
c_time = timeit.timeit(lambda: lib.kan_network(x_ptr, w_ptr, a_ptr, y_ptr), number=n_runs)

# ======= 输出结果 =======
print(f"\n[Benchmark] Run {n_runs} times")
print(f"PyTorch Time: {torch_time:.6f} seconds")
print(f"C LOMA Time:  {c_time:.6f} seconds")

class TorchKANAutograd(nn.Module):
    def __init__(self, weights, alphas):
        super().__init__()
        self.weights = [torch.tensor(w, dtype=torch.float32) for w in weights]
        self.alphas = [torch.tensor(a, dtype=torch.float32) for a in alphas]

    def forward(self, x):
        for w, a in zip(self.weights, self.alphas):
            s = F.linear(x, w)
            out = torch.zeros_like(s)
            for i in range(s.shape[0]):
                s_i = s[i]
                nonlinear = torch.stack([
                    torch.sigmoid(s_i),
                    torch.tanh(s_i),
                    F.relu(s_i),
                    F.leaky_relu(s_i, 0.01),
                    torch.log1p(torch.exp(s_i)),
                    s_i if s_i > 0 else torch.exp(s_i) - 1,
                ])
                out[i] = torch.dot(a[i], nonlinear)
            x = out
        return x

torch_kan_auto = TorchKANAutograd(weights, alphas)

# 输入为需求导的 Tensor
x_torch = torch.tensor(x, dtype=torch.float32, requires_grad=True)
dx = np.array([1.0, 0.0], dtype=np.float32)  # same as in LOMA forward-diff

def run_pytorch_autograd():
    x_torch.grad = None
    y = torch_kan_auto(x_torch)
    grad = torch.autograd.grad(y, x_torch, grad_outputs=torch.tensor([1.0, 0.0]))[0]  # 方向导数
    return y.detach().numpy(), grad.detach().numpy()

# === LOMA Forward-Diff ===
def run_loma_forward_diff():
    lib.d_kan_network(X, W, A, Y)
    return np.array([Y[i].val for i in range(2)]), np.array([Y[i].dval for i in range(2)])

# === 计时对比 ===
n_runs = 1

pytorch_autograd_time = timeit.timeit(lambda: run_pytorch_autograd(), number=n_runs)
loma_fwd_time = timeit.timeit(lambda: run_loma_forward_diff(), number=n_runs)

print("\n[Forward Derivative Benchmark] Run", n_runs, "times")
print(f"PyTorch Autograd Time: {pytorch_autograd_time:.6f} seconds")
print(f"C LOMA ForwardDiff Time: {loma_fwd_time:.6f} seconds")
"""
import statistics

# ======= 多轮运行设置 =======
n_repeat = 100      # 运行轮数
n_runs = 10000      # 每轮的迭代次数

torch_times = []
c_times = []
torch_auto_times = []
loma_fwd_times = []

for _ in range(n_repeat):
    # Torch 推理
    t1 = timeit.timeit(lambda: torch_kan(torch.tensor(x)).detach().numpy(), number=n_runs)
    torch_times.append(t1)

    # C LOMA 推理
    t2 = timeit.timeit(lambda: lib.kan_network(x_ptr, w_ptr, a_ptr, y_ptr), number=n_runs)
    c_times.append(t2)

    # Torch autograd
    t3 = timeit.timeit(lambda: run_pytorch_autograd(), number=n_runs)
    torch_auto_times.append(t3)

    # C LOMA ForwardDiff
    t4 = timeit.timeit(lambda: run_loma_forward_diff(), number=n_runs)
    loma_fwd_times.append(t4)

# ======= 输出统计结果 =======
def print_stats(name, times):
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"{name:30s} | Mean: {mean_time:.6f}s | Std: {std_time:.6f}s | Per run: {mean_time/n_runs*1e6:.2f} µs")

print("\n=== [Multi-Round Runtime Statistics] ===")
print(f"Repeat {n_repeat} times, each with {n_runs} runs")

print_stats("Torch Primal", torch_times)
print_stats("C LOMA Primal", c_times)
print_stats("Torch Autograd (Reverse)", torch_auto_times)
print_stats("C LOMA ForwardDiff", loma_fwd_times)
"""