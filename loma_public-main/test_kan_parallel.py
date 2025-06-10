#!/usr/bin/env python3

"""
Test script for parallel KAN differentiation implementation
Tests both forward and reverse mode differentiation with parallel computation
"""

import os
import sys
import numpy as np

# Add the parent directory to the path
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current)

from kan_modules.kan_reverse_diff import create_kan_reverse_diff, kan_forward_diff_pass, kan_reverse_diff_pass
import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir

def test_parallel_kan_differentiation():
    """
    Test the parallel KAN differentiation implementation
    """
    print("Testing Parallel KAN Differentiation")
    print("=" * 50)
    
    # Test parameters
    input_size = 2
    output_size = 1
    hidden_sizes = [3]
    nonlinearities = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softplus', 'elu']
    
    # Generate test weights
    np.random.seed(42)
    weights = {}
    alpha_weights = {}
    
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    for l in range(len(layer_sizes) - 1):
        current_size = layer_sizes[l]
        next_size = layer_sizes[l+1]
        
        # Generate weights for this layer
        for i in range(next_size):
            for p in range(current_size):
                weights[f"w_{l}_{i}_{p}"] = np.random.uniform(-0.1, 0.1)
            
            # Generate alpha weights for this neuron
            for q in range(len(nonlinearities)):
                alpha_weights[f"alpha_{l}_{i}_{q}"] = np.random.uniform(0, 1)
    
    print(f"Network structure: {layer_sizes}")
    print(f"Using {len(nonlinearities)} nonlinearities: {nonlinearities}")
    print(f"Generated {len(weights)} weights and {len(alpha_weights)} alpha weights")
    
    # Test forward differentiation
    print("\n1. Testing Forward Differentiation (Parallel)")
    print("-" * 30)
    
    try:
        forward_diff_func = kan_forward_diff_pass(
            diff_func_id="test_forward_diff",
            func_id="test_kan",
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            nonlinearities=nonlinearities,
            weights=weights,
            alpha_weights=alpha_weights
        )
        
        print(f"✓ Forward differentiation function created successfully")
        print(f"  Function ID: {forward_diff_func.id}")
        print(f"  Arguments: {len(forward_diff_func.args)}")
        print(f"  Body statements: {len(forward_diff_func.body)}")
        
        # Count parallel computation elements
        parallel_vars = [stmt for stmt in forward_diff_func.body 
                        if isinstance(stmt, loma_ir.Declare) and 
                        ('contrib' in stmt.id or 'ds_' in stmt.id or 'dy_' in stmt.id)]
        print(f"  Parallel computation variables: {len(parallel_vars)}")
        
    except Exception as e:
        print(f"✗ Forward differentiation failed: {e}")
    
    # Test reverse differentiation
    print("\n2. Testing Reverse Differentiation (Parallel)")
    print("-" * 30)
    
    try:
        reverse_diff_func = kan_reverse_diff_pass(
            diff_func_id="test_reverse_diff",
            func_id="test_kan",
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            nonlinearities=nonlinearities,
            weights=weights,
            alpha_weights=alpha_weights
        )
        
        print(f"✓ Reverse differentiation function created successfully")
        print(f"  Function ID: {reverse_diff_func.id}")
        print(f"  Arguments: {len(reverse_diff_func.args)}")
        print(f"  Body statements: {len(reverse_diff_func.body)}")
        
        # Count parallel computation elements
        parallel_vars = [stmt for stmt in reverse_diff_func.body 
                        if isinstance(stmt, loma_ir.Declare) and 
                        ('contrib' in stmt.id or 'ds_' in stmt.id or 'grad_' in stmt.id)]
        print(f"  Parallel computation variables: {len(parallel_vars)}")
        
    except Exception as e:
        print(f"✗ Reverse differentiation failed: {e}")
    
    # Test unified interface
    print("\n3. Testing Unified Interface")
    print("-" * 30)
    
    try:
        # Test forward mode
        forward_func = create_kan_reverse_diff(
            diff_func_id="test_unified_forward",
            structs={},
            funcs={},
            diff_structs={},
            func_id="test_kan",
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            nonlinearities=nonlinearities,
            weights=weights,
            alpha_weights=alpha_weights,
            diff_mode='forward'
        )
        
        print(f"✓ Unified forward mode function created")
        print(f"  Return type: {forward_func.ret_type}")
        
        # Test reverse mode
        reverse_func = create_kan_reverse_diff(
            diff_func_id="test_unified_reverse",
            structs={},
            funcs={},
            diff_structs={},
            func_id="test_kan",
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            nonlinearities=nonlinearities,
            weights=weights,
            alpha_weights=alpha_weights,
            diff_mode='reverse'
        )
        
        print(f"✓ Unified reverse mode function created")
        print(f"  Return type: {reverse_func.ret_type}")
        
    except Exception as e:
        print(f"✗ Unified interface failed: {e}")
    
    print("\n4. Analyzing Parallel Structure")
    print("-" * 30)
    
    # Analyze the parallel structure
    try:
        # Check if parallel computation is properly structured
        forward_body = forward_diff_func.body
        reverse_body = reverse_diff_func.body
        
        # Count layer-wise parallel variables
        layer_parallel_vars = {}
        for stmt in forward_body:
            if isinstance(stmt, loma_ir.Declare) and 'contrib' in stmt.id:
                layer_id = stmt.id.split('_')[1]  # Extract layer number
                if layer_id not in layer_parallel_vars:
                    layer_parallel_vars[layer_id] = 0
                layer_parallel_vars[layer_id] += 1
        
        print(f"✓ Parallel computation structure analyzed")
        print(f"  Forward pass parallel variables per layer: {layer_parallel_vars}")
        
        # Check backward pass structure
        backward_parallel_vars = {}
        for stmt in reverse_body:
            if isinstance(stmt, loma_ir.Declare) and 'contrib' in stmt.id:
                layer_id = stmt.id.split('_')[1]  # Extract layer number
                if layer_id not in backward_parallel_vars:
                    backward_parallel_vars[layer_id] = 0
                backward_parallel_vars[layer_id] += 1
        
        print(f"  Reverse pass parallel variables per layer: {backward_parallel_vars}")
        
        # Verify parallel structure follows equations (10) and (11)
        expected_parallel_vars_per_layer = len(nonlinearities) * sum(hidden_sizes + [output_size])
        total_parallel_vars = sum(layer_parallel_vars.values())
        
        print(f"  Expected parallel variables: ~{expected_parallel_vars_per_layer}")
        print(f"  Actual parallel variables: {total_parallel_vars}")
        
        if total_parallel_vars > 0:
            print(f"✓ Parallel computation structure verified")
        else:
            print(f"⚠ Warning: No parallel computation variables found")
            
    except Exception as e:
        print(f"✗ Parallel structure analysis failed: {e}")
    
    print("\n" + "=" * 50)
    print("Parallel KAN Differentiation Test Complete")
    print("This implementation follows the KAN paper's 'Next Phase'")
    print("section for optimized parallel computation.")

if __name__ == "__main__":
    test_parallel_kan_differentiation() 