import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np

# Add the parent directory to the path so we can import modules from there
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Compile LOMA code that uses kan_layer without requiring explicit KAN_PARAMS
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kan_path = os.path.join(base_dir, "loma_code", "kan_exp.py")
    with open(kan_path) as f:
        _, lib = compiler.compile(f.read(), target="c", output_filename="_code/kan_example")

    # Load compiled model
    f = lib.kan_network

    # Test the LOMA-compiled KAN function
    x0, x1 = 1.0, 2.0
    result = f(x0, x1)
    print(f"KAN result from LOMA for inputs ({x0}, {x1}): {result}")

    # NumPy reference implementation (same weights as in loma_code/kan_example.py)
    from kan_modules.kan import KANNetwork

    # Define same network structure and seeds
    input_size = 2
    output_size = 1
    hidden_sizes = [3]
    num_nonlinearities = 6

    np.random.seed(42)
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    numpy_kan = KANNetwork(layer_sizes, num_nonlinearities=num_nonlinearities)

    # Copy weights from kan_example.py (same seed used)
    for l in range(len(layer_sizes) - 1):
        layer = numpy_kan.layers[l]
        layer.weights = np.random.uniform(-0.1, 0.1, size=(layer.output_size, layer.input_size))
        alphas = np.random.uniform(0, 1, size=(layer.output_size, num_nonlinearities))
        alphas /= np.sum(alphas, axis=1, keepdims=True)
        layer.alpha_weights = alphas

    numpy_result = numpy_kan.forward(np.array([x0, x1]))
    print(f"NumPy KAN result for inputs ({x0}, {x1}): {numpy_result[0]}")