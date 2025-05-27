#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import compiler
import ctypes
import numpy as np

def test_kan_reverse_diff():
    """Test KAN reverse differentiation"""
    
    # Simple test code with KAN reverse differentiation
    loma_code = """
def kan_reverse(x : In[float]) -> float:
    return kan_layer(x)

d_kan_reverse = rev_diff(kan_reverse)
"""
    
    try:
        print("Testing KAN reverse differentiation...")
        
        # Compile the code
        structs, lib = compiler.compile(loma_code,
                                      target='c',
                                      output_filename='_code/test_kan_reverse')
        
        # Test reverse differentiation
        x = 0.5
        _dx = ctypes.c_float(0.0)
        _dreturn = 1.0  # Seed derivative
        
        # Compute derivative
        lib.d_kan_reverse(x, ctypes.byref(_dx), _dreturn)
        
        print(f"Input: x = {x}")
        print(f"Output gradient: _dreturn = {_dreturn}")
        print(f"Input gradient: _dx = {_dx.value}")
        
        # Verify the primal function works
        primal_result = lib.kan_reverse(x)
        print(f"Primal result: kan_reverse({x}) = {primal_result}")
        
        # Check that we got some non-zero gradient
        assert _dx.value != 0.0, "Expected non-zero gradient"
        
        print("✓ KAN reverse differentiation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ KAN reverse differentiation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kan_reverse_diff()
    sys.exit(0 if success else 1) 