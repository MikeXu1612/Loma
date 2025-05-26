import os
import sys

# Add the parent directory to the path so we can import modules from there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import compiler
import parser
from kan_modules.kan import create_kan_in_loma, KANNetwork
from kan_modules import kan_forward_diff
import numpy as np

def kan_example():
    """
    Example showing how to use KAN in Loma.
    Creates a simple KAN network, compiles it, and runs forward pass.
    """
    print("=== KAN Example ===")
    
    # Create a simple KAN network with 2 inputs, 1 output, and 1 hidden layer with 3 nodes
    input_size = 2
    output_size = 1
    hidden_sizes = [3]
    
    # Create weights and alpha weights for demonstration
    np.random.seed(42)  # for reproducibility
    
    # Dictionary to store weights
    weights = {}
    
    # First layer weights (2 inputs -> 3 hidden)
    for i in range(hidden_sizes[0]):
        for p in range(input_size):
            weights[f"w_0_{i}_{p}"] = np.random.uniform(-0.1, 0.1)
    
    # Second layer weights (3 hidden -> 1 output)
    for i in range(output_size):
        for p in range(hidden_sizes[0]):
            weights[f"w_1_{i}_{p}"] = np.random.uniform(-0.1, 0.1)
    
    # Dictionary to store alpha weights
    alpha_weights = {}
    
    # Alpha weights for nonlinearities (6 nonlinearities per node)
    nonlinearities = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softplus', 'elu']
    num_nonlinearities = len(nonlinearities)
    
    # Generate alpha weights for each layer
    for l in range(len(hidden_sizes) + 1):  # hidden layers + output layer
        next_layer_size = hidden_sizes[l] if l < len(hidden_sizes) else output_size
        
        for i in range(next_layer_size):
            # Generate random weights
            alphas = np.random.uniform(0, 1, num_nonlinearities)
            # Normalize to sum to 1
            alphas = alphas / np.sum(alphas)
            
            for q in range(num_nonlinearities):
                alpha_weights[f"alpha_{l}_{i}_{q}"] = alphas[q]
    
    # Create the KAN function in Loma IR
    kan_func = create_kan_in_loma(
        "kan_network",
        input_size,
        output_size,
        hidden_sizes,
        num_nonlinearities=len(nonlinearities)
    )
    
    # Compile the KAN function
    structs = {'_dfloat': None}  # will be filled in by kan_forward_diff
    funcs = {"kan_network": kan_func}
    diff_structs = {}
    
    # Create the forward differentiation function
    diff_func = kan_forward_diff.create_kan_forward_diff(
        "d_kan_network",
        structs,
        funcs,
        diff_structs,
        "kan_network",
        input_size,
        output_size,
        hidden_sizes,
        nonlinearities,
        weights,
        alpha_weights
    )
    
    funcs["d_kan_network"] = diff_func
    
    # Generate Loma code for the functions
    loma_code = """
class _dfloat:
    val: float
    dval: float

def make__dfloat(val: In[float], dval: In[float]) -> _dfloat:
    ret: _dfloat
    ret.val = val
    ret.dval = dval
    return ret

def kan_network(x0: In[float], x1: In[float]) -> float:
    return 0.0
    """
    
    # Compile the code
    ctypes_structs, lib = compiler.compile(
        loma_code,
        target='c',
        output_filename='kan_example'
    )
    
    # Test the KAN function
    x0, x1 = 1.0, 2.0
    result = lib.kan_network(x0, x1)
    print(f"KAN result for inputs ({x0}, {x1}): {result}")
    
    # Also test the numpy implementation for comparison
    # Create and initialize the KAN network in numpy
    numpy_kan = KANNetwork([input_size] + hidden_sizes + [output_size], num_nonlinearities=num_nonlinearities)
    
    # Set the weights manually to match the Loma implementation
    for l in range(len(hidden_sizes) + 1):
        current_layer_size = input_size if l == 0 else hidden_sizes[l-1]
        next_layer_size = hidden_sizes[l] if l < len(hidden_sizes) else output_size
        
        # Set weights
        for i in range(next_layer_size):
            for p in range(current_layer_size):
                numpy_kan.layers[l].weights[i, p] = weights[f"w_{l}_{i}_{p}"]
        
        # Set alpha weights
        for i in range(next_layer_size):
            for q in range(num_nonlinearities):
                numpy_kan.layers[l].alpha_weights[i, q] = alpha_weights[f"alpha_{l}_{i}_{q}"]
    
    # Run the forward pass
    numpy_result = numpy_kan.forward(np.array([x0, x1]))
    print(f"NumPy KAN result for inputs ({x0}, {x1}): {numpy_result[0]}")
    
    print("Done!")

if __name__ == "__main__":
    kan_example() 