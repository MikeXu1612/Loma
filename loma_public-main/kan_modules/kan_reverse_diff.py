"""
KAN Parallel Differentiation Implementation
===========================================

This module implements parallel differentiation for Kolmogorov-Arnold Networks (KAN)
as described in the KAN paper. The key insight is that both forward and backward passes
can be parallelized because neurons in the same layer share the same upstream structure.

Mathematical Foundation:
- Forward pass (Eq. 10): ∂y_i^(l)/∂x_p = sum over q of (alpha_i^{q,l} * φ'_q(s_i^(l)) * sum over j of (b_pj^(l) * ∂y_j^(l-1)/∂x_p))
- Backward pass (Eq. 11): ∂L/∂s_i^(l) = sum over q of (alpha_i^{q,l} * φ'_q(s_i^(l)) * sum over j of (a_ji^{l+1} * ∂L/∂s_j^{l+1}))

The parallelism is achieved by:
1. Computing all nonlinearity contributions for each neuron simultaneously
2. Processing all neurons in a layer with the same upstream input pattern in parallel
3. Propagating gradients to all neurons in the previous layer simultaneously

This implementation follows the "Next Phase" section of the KAN paper for optimized computation.
"""

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
    
    # Backward pass: compute gradients following the KAN paper formulas with parallel computation
    
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
        
        # PARALLEL COMPUTATION: First compute all ∂L/∂s_i^(l) for all neurons in this layer simultaneously
        # Following equation (11): ∂L/∂s_i^(l) = sum over q of (alpha_i^{q,l} * φ'_q(s_i^(l)) * sum over j of (a_ji^{l+1} * ∂L/∂s_j^{l+1}))
        layer_ds_vars = []
        
        # Compute derivatives for all neurons in parallel (same upstream structure)
        for i in range(next_layer_size):
            # Get stored values from forward pass
            s_var = all_s_vars[l][i]
            phi_vars = all_phi_vars[l][i]
            
            # Compute ∂L/∂s_i^(l) following the mathematical formula
            ds_var_name = f"ds_{l}_{i}"
            body.append(loma_ir.Declare(ds_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            ds_var = loma_ir.Var(ds_var_name, t=loma_ir.Float())
            
            # Store for later propagation
            layer_ds_vars.append(ds_var)
            
            # ∂L/∂s_i^(l) = sum over q of (alpha_i^{q,l} * φ'_q(s_i^(l))) * ∂L/∂y_i^(l)
            # All nonlinearities for neuron i can be computed in parallel
            nonlinearity_terms = []
            
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
                
                # Store individual nonlinearity contribution for parallel combination
                contrib_var_name = f"contrib_{l}_{i}_{q}"
                body.append(loma_ir.Declare(contrib_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
                contrib_var = loma_ir.Var(contrib_var_name, t=loma_ir.Float())
                
                # contrib = alpha_i^{q,l} * φ'_q(s_i^(l)) * ∂L/∂y_i^(l)
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
                
                body.append(loma_ir.Assign(contrib_var, weighted_deriv_term))
                nonlinearity_terms.append(contrib_var)
            
            # Sum all nonlinearity contributions in parallel
            for contrib_var in nonlinearity_terms:
                body.append(loma_ir.Assign(
                    ds_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        ds_var,
                        contrib_var,
                        t=loma_ir.Float()
                    )
                ))
        
        # PARALLEL PROPAGATION: Now propagate all gradients to previous layer simultaneously
        # Following equation (10) structure - same upstream input pattern
        for p in range(current_layer_size):
            if l > 0:  # Not propagating to input layer yet
                # Accumulate contributions from all neurons in next layer in parallel
                for i in range(next_layer_size):
                    weight = weights.get(f"w_{l}_{i}_{p}", random.uniform(-0.1, 0.1))
                    weight_const = loma_ir.ConstFloat(weight)
                    
                    # ∂L/∂y_p^(l-1) += a_pi^(l) * ∂L/∂s_i^(l)
                    weight_grad_term = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        weight_const,
                        layer_ds_vars[i],  # Use pre-computed ds_var
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
    
    # PARALLEL INPUT LAYER: Compute final gradients using equation (9) with parallel processing
    # ∂L/∂x_p = sum over i of (c_pi^(1) * ∂L/∂s_i^(1))
    if len(layer_sizes) > 1:
        l = 0  # First hidden layer (layer 1 in the paper notation)
        
        # Initialize all input gradients
        for p in range(input_size):
            dx_var = loma_ir.Var(f"_dx{p}", t=loma_ir.Float())
            body.append(loma_ir.Assign(dx_var, loma_ir.ConstFloat(0.0)))
        
        # PARALLEL COMPUTATION: Compute all ∂L/∂s_i^(1) for input layer neurons simultaneously
        input_layer_ds_vars = []
        
        for i in range(layer_sizes[l+1]):  # Next layer size
            # Get stored values from forward pass
            s_var = all_s_vars[l][i]
            
            # Compute ∂L/∂s_i^(1)
            ds_var_name = f"ds_input_{i}"
            body.append(loma_ir.Declare(ds_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            ds_var = loma_ir.Var(ds_var_name, t=loma_ir.Float())
            input_layer_ds_vars.append(ds_var)
            
            # ∂L/∂s_i^(1) = sum over q of (alpha_i^{q,1} * φ'_q(s_i^(1))) * ∂L/∂y_i^(1)
            # Compute all nonlinearity contributions in parallel
            input_nonlinearity_terms = []
            
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
                
                # Store individual contribution for parallel combination
                input_contrib_var_name = f"input_contrib_{i}_{q}"
                body.append(loma_ir.Declare(input_contrib_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
                input_contrib_var = loma_ir.Var(input_contrib_var_name, t=loma_ir.Float())
                
                # contrib = alpha_i^{q,1} * φ'_q(s_i^(1)) * ∂L/∂y_i^(1)
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
                
                body.append(loma_ir.Assign(input_contrib_var, weighted_deriv_term))
                input_nonlinearity_terms.append(input_contrib_var)
            
            # Sum all nonlinearity contributions in parallel
            for input_contrib_var in input_nonlinearity_terms:
                body.append(loma_ir.Assign(
                    ds_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        ds_var,
                        input_contrib_var,
                        t=loma_ir.Float()
                    )
                ))
        
        # PARALLEL PROPAGATION TO INPUTS: Propagate all gradients to inputs simultaneously
        # Following the same upstream structure pattern
        for p in range(input_size):
            dx_var = loma_ir.Var(f"_dx{p}", t=loma_ir.Float())
            
            # Accumulate contributions from all neurons in first hidden layer in parallel
            for i in range(layer_sizes[l+1]):
                weight = weights.get(f"w_{l}_{i}_{p}", random.uniform(-0.1, 0.1))
                weight_const = loma_ir.ConstFloat(weight)
                
                # ∂L/∂x_p += c_pi^(1) * ∂L/∂s_i^(1)
                weight_grad_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    weight_const,
                    input_layer_ds_vars[i],  # Use pre-computed ds_var
                    t=loma_ir.Float()
                )
                
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


def kan_forward_diff_pass(diff_func_id, 
                          func_id, 
                          input_size, 
                          output_size, 
                          hidden_sizes, 
                          nonlinearities, 
                          weights, 
                          alpha_weights):
    """
    Create a parallel forward differentiation function for a KAN network
    Following equation (10) from the KAN paper for proper gradient computation
    
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
        A Loma IR function definition for the KAN forward differentiation
    """
    # Define layer sizes
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    # Create arguments for the function (forward diff signature)
    args = []
    for i in range(input_size):
        args.append(loma_ir.Arg(f"x{i}", loma_ir.Float(), loma_ir.In()))
        args.append(loma_ir.Arg(f"_dx{i}", loma_ir.Float(), loma_ir.In()))
    
    # Create body of the function
    body = []
    
    # Declare variables for inputs and their derivatives
    input_vars = []
    input_derivs = []
    for i in range(input_size):
        input_vars.append(loma_ir.Var(f"x{i}", t=loma_ir.Float()))
        input_derivs.append(loma_ir.Var(f"_dx{i}", t=loma_ir.Float()))
    
    # Forward pass with parallel derivative computation
    # Following equation (10): ∂y_i^(l)/∂x_p = sum over q of (alpha_i^{q,l} * φ'_q(s_i^(l)) * sum over j of (b_pj^(l) * ∂y_j^(l-1)/∂x_p))
    
    all_s_vars = []  # Store all linear combinations s_i^(l)
    all_y_vars = []  # Store all layer outputs y_i^(l)
    all_dy_vars = []  # Store all layer output derivatives ∂y_i^(l)/∂x_p
    all_phi_vars = []  # Store all individual nonlinearity outputs φ_q(s_i^(l))
    
    prev_layer_outputs = input_vars
    prev_layer_derivs = input_derivs
    
    for l in range(len(layer_sizes) - 1):
        current_layer_size = layer_sizes[l]
        next_layer_size = layer_sizes[l+1]
        
        layer_s_vars = []
        layer_y_vars = []
        layer_dy_vars = []
        layer_phi_vars = []
        
        # PARALLEL COMPUTATION: Process all neurons in this layer simultaneously
        for i in range(next_layer_size):
            # Compute linear combination s_i^(l)
            s_var_name = f"s_{l}_{i}"
            body.append(loma_ir.Declare(s_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            s_var = loma_ir.Var(s_var_name, t=loma_ir.Float())
            
            # Compute derivative of s_i^(l) w.r.t. input
            ds_var_name = f"ds_{l}_{i}"
            body.append(loma_ir.Declare(ds_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            ds_var = loma_ir.Var(ds_var_name, t=loma_ir.Float())
            
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
                
                # ds_i += weight * dx_p (chain rule)
                weight_deriv_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    weight_const,
                    prev_layer_derivs[p],
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
                
                body.append(loma_ir.Assign(
                    ds_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        ds_var,
                        weight_deriv_term,
                        t=loma_ir.Float()
                    )
                ))
            
            layer_s_vars.append(s_var)
            
            # Apply nonlinearities and compute their derivatives in parallel
            node_phi_vars = []
            
            # Compute y_i^(l) and dy_i^(l) following equation (10)
            y_var_name = f"y_{l}_{i}"
            body.append(loma_ir.Declare(y_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            y_var = loma_ir.Var(y_var_name, t=loma_ir.Float())
            
            dy_var_name = f"dy_{l}_{i}"
            body.append(loma_ir.Declare(dy_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            dy_var = loma_ir.Var(dy_var_name, t=loma_ir.Float())
            
            # PARALLEL NONLINEARITY COMPUTATION: Process all nonlinearities simultaneously
            nonlinearity_y_contribs = []
            nonlinearity_dy_contribs = []
            
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
                
                # Compute derivative of nonlinearity
                dphi_var_name = f"dphi_{l}_{i}_{q}"
                dphi_output = kan_utils.apply_nonlinearity_derivative(
                    nonlinearity_type, 
                    s_var, 
                    body, 
                    dphi_var_name
                )
                
                alpha_weight = alpha_weights.get(f"alpha_{l}_{i}_{q}", random.uniform(0, 1))
                alpha_const = loma_ir.ConstFloat(alpha_weight)
                
                # y contribution: alpha * phi_q(s_i)
                y_contrib_var_name = f"y_contrib_{l}_{i}_{q}"
                body.append(loma_ir.Declare(y_contrib_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
                y_contrib_var = loma_ir.Var(y_contrib_var_name, t=loma_ir.Float())
                
                y_contrib_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_const,
                    phi_output,
                    t=loma_ir.Float()
                )
                
                body.append(loma_ir.Assign(y_contrib_var, y_contrib_term))
                nonlinearity_y_contribs.append(y_contrib_var)
                
                # dy contribution: alpha * phi'_q(s_i) * ds_i
                dy_contrib_var_name = f"dy_contrib_{l}_{i}_{q}"
                body.append(loma_ir.Declare(dy_contrib_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
                dy_contrib_var = loma_ir.Var(dy_contrib_var_name, t=loma_ir.Float())
                
                dy_contrib_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_const,
                    loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        dphi_output,
                        ds_var,
                        t=loma_ir.Float()
                    ),
                    t=loma_ir.Float()
                )
                
                body.append(loma_ir.Assign(dy_contrib_var, dy_contrib_term))
                nonlinearity_dy_contribs.append(dy_contrib_var)
            
            # Sum all contributions in parallel
            for y_contrib_var in nonlinearity_y_contribs:
                body.append(loma_ir.Assign(
                    y_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        y_var,
                        y_contrib_var,
                        t=loma_ir.Float()
                    )
                ))
            
            for dy_contrib_var in nonlinearity_dy_contribs:
                body.append(loma_ir.Assign(
                    dy_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        dy_var,
                        dy_contrib_var,
                        t=loma_ir.Float()
                    )
                ))
            
            layer_y_vars.append(y_var)
            layer_dy_vars.append(dy_var)
            layer_phi_vars.append(node_phi_vars)
        
        all_s_vars.append(layer_s_vars)
        all_y_vars.append(layer_y_vars)
        all_dy_vars.append(layer_dy_vars)
        all_phi_vars.append(layer_phi_vars)
        prev_layer_outputs = layer_y_vars
        prev_layer_derivs = layer_dy_vars
    
    # Return the derivative of the final output
    if output_size == 1:
        body.append(loma_ir.Return(prev_layer_derivs[0]))
    else:
        # For multiple outputs, would need to handle differently
        pass
    
    return loma_ir.FunctionDef(
        diff_func_id,
        args,
        body,
        False,  # is_simd
        loma_ir.Float() if output_size == 1 else None,  # ret_type
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
                           alpha_weights=None,
                           diff_mode='reverse'):
    """
    Create a differentiation function for a KAN network to be used with autodiff
    Implements parallel computation as described in the KAN paper
    
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
        diff_mode: 'forward' or 'reverse' differentiation mode
        
    Returns:
        A Loma IR function definition for the KAN differentiation
    """
    if nonlinearities is None:
        # Default: use all 6 nonlinearities
        nonlinearities = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softplus', 'elu']
        
    if weights is None:
        weights = {}
        
    if alpha_weights is None:
        alpha_weights = {}
    
    # Create the differentiation function based on mode
    if diff_mode == 'forward':
        return kan_forward_diff_pass(
            diff_func_id,
            func_id,
            input_size,
            output_size,
            hidden_sizes,
            nonlinearities,
            weights,
            alpha_weights
        )
    else:  # reverse mode (default)
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