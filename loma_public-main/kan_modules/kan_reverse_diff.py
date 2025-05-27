import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
from . import kan_utils
import random
import string

def kan_reverse_diff_pass(diff_func_id, 
                          func_id, 
                          input_size, 
                          output_size, 
                          hidden_sizes, 
                          nonlinearities, 
                          weights, 
                          alpha_weights):
    """
    Create a reverse differentiation function for a KAN network
    
    Args:
        diff_func_id: The ID of the differentiated function to create
        func_id: The ID of the original function
        input_size: Size of the input layer
        output_size: Size of the output layer
        hidden_sizes: List of hidden layer sizes
        nonlinearities: List of nonlinearity types for each layer
        weights: Dictionary of weight matrices for each layer
        alpha_weights: Dictionary of alpha weights for each layer
        
    Returns:
        A Loma IR function definition for the KAN reverse differentiation
    """
    # Define layer sizes
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    # Create arguments for the function (reverse diff signature)
    args = []
    for i in range(input_size):
        args.append(loma_ir.Arg(f"x{i}", loma_ir.Float(), loma_ir.In()))
        args.append(loma_ir.Arg(f"_dx{i}", loma_ir.Float(), loma_ir.Out()))
    
    # Add the return gradient argument
    args.append(loma_ir.Arg("_dreturn", loma_ir.Float(), loma_ir.In()))
    
    # Create body of the function
    body = []
    
    # Forward pass: compute and store intermediate values
    # We need to store all intermediate values for the backward pass
    
    # Declare variables for inputs
    input_vars = []
    for i in range(input_size):
        var_name = f"x{i}"
        input_vars.append(loma_ir.Var(var_name, t=loma_ir.Float()))
    
    # Forward pass: compute all intermediate values and store them
    all_s_vars = []  # Store all linear combinations s_i^(l)
    all_y_vars = []  # Store all layer outputs y_i^(l)
    all_nonlinear_vars = []  # Store all nonlinearity outputs
    
    prev_layer_outputs = input_vars
    
    for l in range(len(layer_sizes) - 1):
        current_layer_size = layer_sizes[l]
        next_layer_size = layer_sizes[l+1]
        
        layer_s_vars = []
        layer_y_vars = []
        layer_nonlinear_vars = []
        
        for i in range(next_layer_size):
            # Compute linear combination s_i^(l)
            s_var_name = f"s_{l}_{i}"
            body.append(loma_ir.Declare(s_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            s_var = loma_ir.Var(s_var_name, t=loma_ir.Float())
            
            # Use the provided weights for this layer
            for p in range(current_layer_size):
                weight = weights.get(f"w_{l}_{i}_{p}", random.uniform(-0.1, 0.1))
                weight_const = loma_ir.ConstFloat(weight)
                
                # s_i += weight * x_p
                weight_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    weight_const,
                    prev_layer_outputs[p],
                    t=loma_ir.Float()
                )
                
                body.append(loma_ir.Assign(
                    s_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        s_var,
                        weight_term,
                        t=loma_ir.Float()
                    )
                ))
            
            layer_s_vars.append(s_var)
            
            # Apply nonlinearities and combine with alpha weights
            y_var_name = f"y_{l}_{i}"
            body.append(loma_ir.Declare(y_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            y_var = loma_ir.Var(y_var_name, t=loma_ir.Float())
            
            node_nonlinear_vars = []
            
            # Use the provided alpha weights
            for q, nonlinearity_type in enumerate(nonlinearities):
                alpha_weight = alpha_weights.get(f"alpha_{l}_{i}_{q}", random.uniform(0, 1))
                alpha_const = loma_ir.ConstFloat(alpha_weight)
                
                # Apply nonlinearity q to s_i
                nonlinear_var_name = f"{nonlinearity_type}_{l}_{i}_{q}"
                nonlinear_output = kan_utils.apply_nonlinearity(
                    nonlinearity_type, 
                    s_var, 
                    body, 
                    nonlinear_var_name
                )
                
                node_nonlinear_vars.append((nonlinearity_type, nonlinear_output, alpha_const))
                
                # y_i += alpha * nonlinear_output
                weighted_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_const,
                    nonlinear_output,
                    t=loma_ir.Float()
                )
                
                body.append(loma_ir.Assign(
                    y_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        y_var,
                        weighted_term,
                        t=loma_ir.Float()
                    )
                ))
            
            layer_y_vars.append(y_var)
            layer_nonlinear_vars.append(node_nonlinear_vars)
        
        all_s_vars.append(layer_s_vars)
        all_y_vars.append(layer_y_vars)
        all_nonlinear_vars.append(layer_nonlinear_vars)
        prev_layer_outputs = layer_y_vars
    
    # Backward pass: compute gradients
    # Initialize gradient variables for each layer
    
    # Start with the output gradient
    output_grad_name = "_dreturn"
    
    # Work backwards through the layers
    current_grad = [loma_ir.Var(output_grad_name, t=loma_ir.Float())]
    
    for l in reversed(range(len(layer_sizes) - 1)):
        current_layer_size = layer_sizes[l]
        next_layer_size = layer_sizes[l+1]
        
        # Gradients for the current layer inputs
        prev_grad = []
        for p in range(current_layer_size):
            grad_var_name = f"grad_{l}_{p}"
            body.append(loma_ir.Declare(grad_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            prev_grad.append(loma_ir.Var(grad_var_name, t=loma_ir.Float()))
        
        # For each node in the next layer
        for i in range(next_layer_size):
            # Get the gradient flowing into this node
            node_grad = current_grad[i]
            
            # Get stored values from forward pass
            s_var = all_s_vars[l][i]
            node_nonlinear_vars = all_nonlinear_vars[l][i]
            
            # Compute gradient w.r.t. s_i^(l)
            ds_var_name = f"ds_{l}_{i}"
            body.append(loma_ir.Declare(ds_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            ds_var = loma_ir.Var(ds_var_name, t=loma_ir.Float())
            
            # ds_i = sum over q of (alpha_q * derivative_q(s_i) * dy_i)
            for q, (nonlinearity_type, nonlinear_output, alpha_const) in enumerate(node_nonlinear_vars):
                # Get the derivative of the nonlinearity
                derivative_var_name = f"{nonlinearity_type}_deriv_back_{l}_{i}_{q}"
                derivative_output = kan_utils.apply_nonlinearity_derivative(
                    nonlinearity_type, 
                    s_var, 
                    body, 
                    derivative_var_name
                )
                
                # ds_i += alpha * derivative * dy_i
                deriv_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_const,
                    derivative_output,
                    t=loma_ir.Float()
                )
                
                weighted_deriv_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    deriv_term,
                    node_grad,
                    t=loma_ir.Float()
                )
                
                body.append(loma_ir.Assign(
                    ds_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        ds_var,
                        weighted_deriv_term,
                        t=loma_ir.Float()
                    )
                ))
            
            # Propagate gradient to previous layer inputs
            # dx_p += weight * ds_i
            for p in range(current_layer_size):
                weight = weights.get(f"w_{l}_{i}_{p}", random.uniform(-0.1, 0.1))
                weight_const = loma_ir.ConstFloat(weight)
                
                weight_grad_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    weight_const,
                    ds_var,
                    t=loma_ir.Float()
                )
                
                body.append(loma_ir.Assign(
                    prev_grad[p],
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        prev_grad[p],
                        weight_grad_term,
                        t=loma_ir.Float()
                    )
                ))
        
        current_grad = prev_grad
    
    # Assign final gradients to output arguments
    for i in range(input_size):
        dx_var = loma_ir.Var(f"_dx{i}", t=loma_ir.Float())
        body.append(loma_ir.Assign(dx_var, current_grad[i]))
    
    return loma_ir.FunctionDef(
        diff_func_id,
        args,
        body,
        False,  # is_simd
        None,   # ret_type (void for reverse diff)
    )


def generate_random_string(length=6):
    """Generate a random string of fixed length"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def create_kan_reverse_diff(diff_func_id,
                           structs,
                           funcs,
                           diff_structs,
                           func_id,
                           input_size,
                           output_size,
                           hidden_sizes=[10],
                           nonlinearities=None,
                           weights=None,
                           alpha_weights=None):
    """
    Create a reverse differentiation function for a KAN network to be used with autodiff
    
    Args:
        diff_func_id: The ID of the differentiated function to create
        structs: Dictionary of struct definitions
        funcs: Dictionary of function definitions
        diff_structs: Dictionary of differential struct definitions
        func_id: The ID of the original function
        input_size: Size of the input layer
        output_size: Size of the output layer
        hidden_sizes: List of hidden layer sizes
        nonlinearities: List of nonlinearity types for each layer
        weights: Dictionary of weight matrices for each layer
        alpha_weights: Dictionary of alpha weights for each layer
        
    Returns:
        A Loma IR function definition for the KAN reverse differentiation
    """
    if nonlinearities is None:
        # Default: use all 6 nonlinearities
        nonlinearities = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softplus', 'elu']
        
    if weights is None:
        weights = {}
        
    if alpha_weights is None:
        alpha_weights = {}
    
    # Create the reverse differentiation function
    return kan_reverse_diff_pass(
        diff_func_id,
        func_id,
        input_size,
        output_size,
        hidden_sizes,
        nonlinearities,
        weights,
        alpha_weights
    ) 