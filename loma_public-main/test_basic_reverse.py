#!/usr/bin/env python3

"""
Basic Test for reverse differentiation
Simple test case: f(x) = x^2, so f'(x) = 2x
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import compiler
import ctypes
import numpy as np

def test_basic_reverse_diff():
    """Test basic reverse differentiation with f(x) = x^2"""
    
    # Simple loma code for f(x) = x^2
    loma_code = """
def square(x : In[float]) -> float:
    return x * x

d_square = rev_diff(square)
"""
    
    try:
        print("Testing Basic Reverse Differentiation...")
        print("=" * 50)
        print("Function: f(x) = x²")
        print("Expected derivative: f'(x) = 2x")
        
        # Compile the code
        structs, lib = compiler.compile(loma_code,
                                      target='c',
                                      output_filename='_code/test_basic_reverse')
        
        print("✓ Code compiled successfully")
        
        # Test reverse differentiation
        test_inputs = [0.0, 1.0, 2.0, -1.0, -2.0, 0.5]
        
        print("\nTesting reverse differentiation:")
        print("-" * 40)
        print("       x         f(x)        df/dx     Expected")
        print("-" * 40)
        
        all_passed = True
        
        for x in test_inputs:
            # Create mutable reference for gradient
            _dx = ctypes.c_float(0.0)
            _dreturn = 1.0  # Seed derivative
            
            # Compute derivative
            lib.d_square(x, ctypes.byref(_dx), _dreturn)
            
            # Compute primal function value
            primal_result = lib.square(x)
            
            # Expected values
            expected_f = x * x
            expected_df = 2 * x
            
            # Check results
            f_error = abs(primal_result - expected_f)
            df_error = abs(_dx.value - expected_df)
            
            status = "✓" if (f_error < 1e-6 and df_error < 1e-6) else "✗"
            if status == "✗":
                all_passed = False
            
            print(f"{x:8.2f} {primal_result:12.6f} {_dx.value:12.6f} {expected_df:12.6f} {status}")
        
        print("-" * 40)
        
        if all_passed:
            print("✓ All basic reverse differentiation tests passed!")
        else:
            print("✗ Some tests failed!")
            return False
        
        # Numerical gradient verification
        print("\nNumerical gradient verification:")
        print("-" * 30)
        
        def numerical_gradient(f, x, h=1e-5):
            """Compute numerical gradient using finite differences"""
            return (f(x + h) - f(x - h)) / (2 * h)
        
        def eval_function(x):
            """Evaluate the square function"""
            return lib.square(x)
        
        for x in [1.0, 2.0, -1.5]:
            # Analytical gradient
            _dx = ctypes.c_float(0.0)
            lib.d_square(x, ctypes.byref(_dx), 1.0)
            analytical_grad = _dx.value
            
            # Numerical gradient
            numerical_grad = numerical_gradient(eval_function, x)
            
            # Compare
            error = abs(analytical_grad - numerical_grad)
            relative_error = error / (abs(numerical_grad) + 1e-8)
            
            print(f"x = {x:4.1f}: analytical = {analytical_grad:8.4f}, numerical = {numerical_grad:8.4f}, error = {relative_error*100:.3f}%")
            
            # Check if gradients match within tolerance
            if relative_error > 0.01:  # 1% tolerance
                print(f"✗ Gradient mismatch at x={x}")
                return False
        
        print("✓ Numerical gradient verification passed!")
        
        print("\n" + "=" * 50)
        print("✓ Basic Reverse Differentiation Test Complete!")
        print("✓ Function: f(x) = x²")
        print("✓ Derivative: f'(x) = 2x")
        print("✓ All tests passed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic reverse differentiation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_reverse_diff()
    if success:
        print("\n🎉 Basic reverse differentiation test completed successfully!")
    else:
        print("\n❌ Basic reverse differentiation test failed!")
        sys.exit(1) 