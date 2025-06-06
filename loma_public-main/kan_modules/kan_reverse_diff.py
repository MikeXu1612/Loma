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
    Following the mathematical formulas from the KAN paper for proper gradient computation
    
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
    all_phi_vars = []  # Store all individual nonlinearity outputs φ_q(s_i^(l))
    
    prev_layer_outputs = input_vars
    
    for l in range(len(layer_sizes) - 1):
        current_layer_size = layer_sizes[l]
        next_layer_size = layer_sizes[l+1]
        
        layer_s_vars = []
        layer_y_vars = []
        layer_phi_vars = []
        
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
            
            # Apply nonlinearities and store individual outputs
            node_phi_vars = []
            
            # Store individual nonlinearity outputs φ_q(s_i^(l))
            for q, nonlinearity_type in enumerate(nonlinearities):
                # Apply nonlinearity q to s_i
                phi_var_name = f"phi_{l}_{i}_{q}"
                phi_output = kan_utils.apply_nonlinearity(
                    nonlinearity_type, 
                    s_var, 
                    body, 
                    phi_var_name
                )
                node_phi_vars.append(phi_output)
            
            # Compute y_i^(l) = sum over q of (alpha_i^{q,l} * φ_q(s_i^(l)))
            y_var_name = f"y_{l}_{i}"
            body.append(loma_ir.Declare(y_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            y_var = loma_ir.Var(y_var_name, t=loma_ir.Float())
            
            # Use the provided alpha weights
            for q, nonlinearity_type in enumerate(nonlinearities):
                alpha_weight = alpha_weights.get(f"alpha_{l}_{i}_{q}", random.uniform(0, 1))
                alpha_const = loma_ir.ConstFloat(alpha_weight)
                
                # y_i += alpha * phi_q(s_i)
                weighted_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_const,
                    node_phi_vars[q],
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
            layer_phi_vars.append(node_phi_vars)
        
        all_s_vars.append(layer_s_vars)
        all_y_vars.append(layer_y_vars)
        all_phi_vars.append(layer_phi_vars)
        prev_layer_outputs = layer_y_vars
    
    # Backward pass: compute gradients following the KAN paper formulas
    
    # Start with the output gradient (equation 4: ∂L/∂y_i^(L) = ∂L/∂f)
    output_grad_name = "_dreturn"
    
    # Initialize gradient arrays for each layer
    # grad_y[l][i] stores ∂L/∂y_i^(l)
    grad_y = []
    for l in range(len(layer_sizes)):
        layer_grad = []
        for i in range(layer_sizes[l]):
            if l == len(layer_sizes) - 1:  # Output layer
                if i == 0:  # Assuming single output
                    layer_grad.append(loma_ir.Var(output_grad_name, t=loma_ir.Float()))
                else:
                    layer_grad.append(loma_ir.ConstFloat(0.0))
            else:
                grad_var_name = f"grad_y_{l}_{i}"
                body.append(loma_ir.Declare(grad_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
                layer_grad.append(loma_ir.Var(grad_var_name, t=loma_ir.Float()))
        grad_y.append(layer_grad)
    
    # Work backwards through the layers (excluding input layer)
    for l in reversed(range(len(layer_sizes) - 1)):
        current_layer_size = layer_sizes[l]
        next_layer_size = layer_sizes[l+1]
        
        # For each node in the current layer (l)
        for i in range(current_layer_size):
            # Reset gradient for this node
            if l > 0:  # Not input layer
                body.append(loma_ir.Assign(grad_y[l][i], loma_ir.ConstFloat(0.0)))
        
        # Compute gradients using equation (6): ∂L/∂s_i^(l) = sum over q of (alpha_i^{q,l} * φ'_q(s_i^(l)) * sum over j of (a_ji^{l+1} * ∂L/∂s_j^{l+1}))
        for i in range(next_layer_size):
            # Get stored values from forward pass
            s_var = all_s_vars[l][i]
            phi_vars = all_phi_vars[l][i]
            
            # Compute ∂L/∂s_i^(l) following the mathematical formula
            ds_var_name = f"ds_{l}_{i}"
            body.append(loma_ir.Declare(ds_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            ds_var = loma_ir.Var(ds_var_name, t=loma_ir.Float())
            
            # ∂L/∂s_i^(l) = sum over q of (alpha_i^{q,l} * φ'_q(s_i^(l))) * ∂L/∂y_i^(l)
            for q, nonlinearity_type in enumerate(nonlinearities):
                alpha_weight = alpha_weights.get(f"alpha_{l}_{i}_{q}", random.uniform(0, 1))
                alpha_const = loma_ir.ConstFloat(alpha_weight)
                
                # Get the derivative of the nonlinearity φ'_q(s_i^(l))
                derivative_var_name = f"{nonlinearity_type}_deriv_back_{l}_{i}_{q}"
                derivative_output = kan_utils.apply_nonlinearity_derivative(
                    nonlinearity_type, 
                    s_var, 
                    body, 
                    derivative_var_name
                )
                
                # ∂L/∂s_i^(l) += alpha_i^{q,l} * φ'_q(s_i^(l)) * ∂L/∂y_i^(l)
                alpha_deriv_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_const,
                    derivative_output,
                    t=loma_ir.Float()
                )
                
                weighted_deriv_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_deriv_term,
                    grad_y[l+1][i],  # ∂L/∂y_i^(l) 
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
            
            # Propagate gradient to previous layer using equation (5): ∂L/∂s_j^(l-1) = sum over i of (a_pi^(l) * ∂L/∂s_i^(l))
            for p in range(current_layer_size):
                if l > 0:  # Not propagating to input layer yet
                    weight = weights.get(f"w_{l}_{i}_{p}", random.uniform(-0.1, 0.1))
                    weight_const = loma_ir.ConstFloat(weight)
                    
                    # ∂L/∂y_p^(l-1) += a_pi^(l) * ∂L/∂s_i^(l)
                    weight_grad_term = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        weight_const,
                        ds_var,
                        t=loma_ir.Float()
                    )
                    
                    body.append(loma_ir.Assign(
                        grad_y[l][p],
                        loma_ir.BinaryOp(
                            loma_ir.Add(),
                            grad_y[l][p],
                            weight_grad_term,
                            t=loma_ir.Float()
                        )
                    ))
    
    # For the input layer, compute final gradients using equation (9): ∂L/∂x_p = sum over i of (c_pi^(1) * ∂L/∂s_i^(1))
    if len(layer_sizes) > 1:
        l = 0  # First hidden layer (layer 1 in the paper notation)
        for i in range(layer_sizes[l+1]):  # Next layer size
            # Get stored values from forward pass
            s_var = all_s_vars[l][i]
            
            # Compute ∂L/∂s_i^(1)
            ds_var_name = f"ds_input_{i}"
            body.append(loma_ir.Declare(ds_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            ds_var = loma_ir.Var(ds_var_name, t=loma_ir.Float())
            
            # ∂L/∂s_i^(1) = sum over q of (alpha_i^{q,1} * φ'_q(s_i^(1))) * ∂L/∂y_i^(1)
            for q, nonlinearity_type in enumerate(nonlinearities):
                alpha_weight = alpha_weights.get(f"alpha_{l}_{i}_{q}", random.uniform(0, 1))
                alpha_const = loma_ir.ConstFloat(alpha_weight)
                
                # Get the derivative of the nonlinearity φ'_q(s_i^(1))
                derivative_var_name = f"{nonlinearity_type}_deriv_input_{i}_{q}"
                derivative_output = kan_utils.apply_nonlinearity_derivative(
                    nonlinearity_type, 
                    s_var, 
                    body, 
                    derivative_var_name
                )
                
                # ∂L/∂s_i^(1) += alpha_i^{q,1} * φ'_q(s_i^(1)) * ∂L/∂y_i^(1)
                alpha_deriv_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_const,
                    derivative_output,
                    t=loma_ir.Float()
                )
                
                weighted_deriv_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_deriv_term,
                    grad_y[l+1][i],
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
            
            # Propagate to input: ∂L/∂x_p += c_pi^(1) * ∂L/∂s_i^(1)
            for p in range(input_size):
                weight = weights.get(f"w_{l}_{i}_{p}", random.uniform(-0.1, 0.1))
                weight_const = loma_ir.ConstFloat(weight)
                
                dx_var = loma_ir.Var(f"_dx{p}", t=loma_ir.Float())
                
                weight_grad_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    weight_const,
                    ds_var,
                    t=loma_ir.Float()
                )
                
                # Initialize if first time
                if i == 0:
                    body.append(loma_ir.Assign(dx_var, loma_ir.ConstFloat(0.0)))
                
                body.append(loma_ir.Assign(
                    dx_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        dx_var,
                        weight_grad_term,
                        t=loma_ir.Float()
                    )
                ))
    
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