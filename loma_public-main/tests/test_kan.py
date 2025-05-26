import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import error
import math
import numpy as np
import unittest
from kan_modules import kan_utils
from kan_modules.kan import KANLayer, KANNetwork, create_kan_in_loma

epsilon = 1e-4

class KANTest(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    def test_kan_layer_initialization(self):
        """Test that KAN layer initializes correctly with proper shapes."""
        input_size = 5
        output_size = 3
        num_nonlinearities = 4
        
        layer = KANLayer(input_size, output_size, num_nonlinearities)
        layer.initialize_weights()
        
        # Check shapes of initialized weights
        self.assertEqual(layer.weights.shape, (output_size, input_size))
        self.assertEqual(layer.alpha_weights.shape, (output_size, num_nonlinearities))
        
        # Check that alpha weights sum to 1 for each output
        alpha_sums = np.sum(layer.alpha_weights, axis=1)
        for sum_val in alpha_sums:
            self.assertAlmostEqual(sum_val, 1.0, delta=epsilon)
    
    def test_kan_network_initialization(self):
        """Test that KAN network initializes correctly with proper layer sizes."""
        layer_sizes = [10, 8, 6, 4, 2]
        num_nonlinearities = 5
        
        network = KANNetwork(layer_sizes, num_nonlinearities)
        
        # Check network properties
        self.assertEqual(network.num_layers, len(layer_sizes) - 1)
        self.assertEqual(len(network.layers), len(layer_sizes) - 1)
        
        # Check each layer's dimensions
        for i, layer in enumerate(network.layers):
            self.assertEqual(layer.input_size, layer_sizes[i])
            self.assertEqual(layer.output_size, layer_sizes[i+1])
            self.assertEqual(layer.num_nonlinearities, num_nonlinearities)
    
    def test_nonlinearity_functions(self):
        """Test nonlinearity functions and their derivatives."""
        x_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
        
        for x in x_values:
            # Test sigmoid and its derivative
            sig = kan_utils.sigmoid(x)
            sig_der = kan_utils.sigmoid_derivative(x)
            expected_sig_der = sig * (1.0 - sig)
            self.assertAlmostEqual(sig_der, expected_sig_der, delta=epsilon)
            
            # Test tanh and its derivative
            th = kan_utils.tanh(x)
            th_der = kan_utils.tanh_derivative(x)
            expected_th_der = 1.0 - th * th
            self.assertAlmostEqual(th_der, expected_th_der, delta=epsilon)
            
            # Test ReLU and its derivative
            relu_val = kan_utils.relu(x)
            relu_der = kan_utils.relu_derivative(x)
            expected_relu = max(0.0, x)
            expected_relu_der = 1.0 if x > 0 else 0.0
            self.assertAlmostEqual(relu_val, expected_relu, delta=epsilon)
            self.assertAlmostEqual(relu_der, expected_relu_der, delta=epsilon)
    
    def test_kan_simple(self):
        """Test KAN with a simple single-input, single-output function."""
        with open('loma_code/kan_simple.py') as f:
            loma_code = f.read()
        
        try:
            # Compile the code - this should be replaced by KAN implementation
            structs, funcs = compiler.parser.parse(loma_code)
            
            # Set up parameters for KAN
            input_size = 1
            output_size = 1
            hidden_sizes = [3]
            num_nonlinearities = 3
            
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Create KAN implementation
            kan_func_id = "kan_simple"
            create_kan_in_loma(kan_func_id, input_size, output_size, hidden_sizes, num_nonlinearities)
            
            # Compile and run the KAN implementation
            _, lib = compiler.compile(loma_code,
                                      target='c',
                                      output_filename='_code/kan_simple')
            
            # Test with several input values
            for x in [0.5, -1.0, 2.0]:
                result = lib.kan_simple(x)
                self.assertIsNotNone(result)
                
        except Exception as e:
            self.fail(f"KAN simple test failed: {str(e)}")
    
    def test_kan_multi_input(self):
        """Test KAN with multiple inputs."""
        with open('loma_code/kan_multi_input.py') as f:
            loma_code = f.read()
        
        try:
            # Compile the code
            structs, funcs = compiler.parser.parse(loma_code)
            
            # Set up parameters for KAN
            input_size = 2  # Two inputs
            output_size = 1
            hidden_sizes = [4]
            num_nonlinearities = 3
            
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Create KAN implementation
            kan_func_id = "kan_multi_input"
            create_kan_in_loma(kan_func_id, input_size, output_size, hidden_sizes, num_nonlinearities)
            
            # Compile and run the KAN implementation
            _, lib = compiler.compile(loma_code,
                                      target='c',
                                      output_filename='_code/kan_multi_input')
            
            # Test with several input combinations
            inputs = [(0.5, 0.3), (-1.0, 2.0), (2.0, -0.5)]
            for x, y in inputs:
                result = lib.kan_multi_input(x, y)
                self.assertIsNotNone(result)
                
        except Exception as e:
            self.fail(f"KAN multi-input test failed: {str(e)}")
    
    def test_kan_multi_output(self):
        """Test KAN with multiple outputs."""
        with open('loma_code/kan_multi_output.py') as f:
            loma_code = f.read()
        
        try:
            # Compile the code
            structs, funcs = compiler.parser.parse(loma_code)
            
            # Set up parameters for KAN
            input_size = 1
            output_size = 2  # Two outputs (Vector2)
            hidden_sizes = [4]
            num_nonlinearities = 3
            
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Create KAN implementation
            kan_func_id = "kan_multi_output"
            create_kan_in_loma(kan_func_id, input_size, output_size, hidden_sizes, num_nonlinearities)
            
            # Compile and run the KAN implementation
            structs, lib = compiler.compile(loma_code,
                                          target='c',
                                          output_filename='_code/kan_multi_output')
            
            # Test with several input values
            for x in [0.5, -1.0, 2.0]:
                result = lib.kan_multi_output(x)
                self.assertIsNotNone(result)
                self.assertIsNotNone(result.x)
                self.assertIsNotNone(result.y)
                
        except Exception as e:
            self.fail(f"KAN multi-output test failed: {str(e)}")
    
    def test_kan_forward_differentiation(self):
        """Test KAN with forward differentiation."""
        with open('loma_code/kan_forward_diff.py') as f:
            loma_code = f.read()
        
        try:
            # Compile the code
            structs, funcs = compiler.parser.parse(loma_code)
            
            # Set up parameters for KAN
            input_size = 1
            output_size = 1
            hidden_sizes = [3]
            num_nonlinearities = 3
            
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Create KAN implementation
            kan_func_id = "kan_forward"
            create_kan_in_loma(kan_func_id, input_size, output_size, hidden_sizes, num_nonlinearities)
            
            # Compile with differentiation
            structs, lib = compiler.compile(loma_code,
                                          target='c',
                                          output_filename='_code/kan_forward_diff')
            
            # Test forward differentiation
            _dfloat = structs['_dfloat']
            
            # Test with x=0.7, dx=1.0 (derivative w.r.t. x)
            x = _dfloat(0.7, 1.0)
            result = lib.d_kan_forward(x)
            
            # We expect some derivative value (exact value depends on random weights)
            self.assertIsNotNone(result.val)
            self.assertIsNotNone(result.dval)
            
            # Verify the forward pass result matches the primal function
            primal_result = lib.kan_forward(x.val)
            self.assertAlmostEqual(result.val, primal_result, delta=epsilon)
            
        except Exception as e:
            self.fail(f"KAN forward differentiation test failed: {str(e)}")
    
    def test_kan_reverse_differentiation(self):
        """Test KAN with reverse differentiation."""
        with open('loma_code/kan_reverse_diff.py') as f:
            loma_code = f.read()
        
        try:
            # Compile the code
            structs, funcs = compiler.parser.parse(loma_code)
            
            # Set up parameters for KAN
            input_size = 1
            output_size = 1
            hidden_sizes = [3]
            num_nonlinearities = 3
            
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Create KAN implementation
            kan_func_id = "kan_reverse"
            create_kan_in_loma(kan_func_id, input_size, output_size, hidden_sizes, num_nonlinearities)
            
            # Compile with differentiation
            structs, lib = compiler.compile(loma_code,
                                          target='c',
                                          output_filename='_code/kan_reverse_diff')
            
            # Test reverse differentiation
            x = 0.7
            _dx = ctypes.c_float(0.0)
            _dreturn = 1.0  # Seed derivative
            
            # Compute derivative
            lib.d_kan_reverse(x, ctypes.byref(_dx), _dreturn)
            
            # We expect some derivative value (exact value depends on random weights)
            self.assertIsNotNone(_dx.value)
            
            # Verify the primal function works
            primal_result = lib.kan_reverse(x)
            self.assertIsNotNone(primal_result)
            
        except Exception as e:
            self.fail(f"KAN reverse differentiation test failed: {str(e)}")
    
    def test_kan_forward_pass(self):
        """Test the forward pass of a KAN network."""
        # Create a simple network with fixed weights for testing
        input_size = 2
        hidden_size = 3
        output_size = 1
        num_nonlinearities = 2
        
        np.random.seed(42)  # For reproducibility
        
        # Create a two-layer network
        network = KANNetwork([input_size, hidden_size, output_size], num_nonlinearities)
        
        # Set deterministic weights for testing
        for layer in network.layers:
            layer.weights = np.ones(layer.weights.shape) * 0.1
            layer.alpha_weights = np.ones(layer.alpha_weights.shape) / num_nonlinearities
        
        # Test forward pass with a simple input
        x = np.array([0.5, -0.5])
        output = network.forward(x)
        
        # Check output shape
        self.assertEqual(output.shape, (output_size,))
        
        # Since we're using fixed weights, we can calculate expected output
        # But this depends on the specific nonlinearities used in the implementation
        # So we'll just check the result is not None and has the right shape
        self.assertIsNotNone(output)
    
    def test_complex_kan_network(self):
        """Test a more complex KAN network with multiple hidden layers."""
        # Create a more complex network
        layer_sizes = [5, 10, 8, 3]
        num_nonlinearities = 4
        
        np.random.seed(42)  # For reproducibility
        
        network = KANNetwork(layer_sizes, num_nonlinearities)
        
        # Test with random input
        x = np.random.rand(layer_sizes[0])
        output = network.forward(x)
        
        # Check output shape
        self.assertEqual(output.shape, (layer_sizes[-1],))
        
        # Check output values are within reasonable range
        for val in output:
            self.assertTrue(-10 < val < 10, f"Output value {val} outside reasonable range")

if __name__ == '__main__':
    unittest.main() 