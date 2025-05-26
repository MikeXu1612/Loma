import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
from . import kan_utils
import random
import string

def kan_forward_diff_pass(diff_func_id, 
                          func_id, 
                          input_size, 
                          output_size, 
                          hidden_sizes, 
                          nonlinearities, 
                          weights, 
                          alpha_weights):
    """
    Create a forward differentiation function for a KAN network
    
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
    
    # Create arguments for the function
    args = []
    for i in range(input_size):
        args.append(loma_ir.Arg(f"x{i}", loma_ir.Struct("_dfloat", []), loma_ir.In()))
    
    # Create body of the function
    body = []
    
    # Declare variables for inputs
    input_vars = []
    input_dvals = []
    for i in range(input_size):
        var_name = f"x{i}"
        # Access the value and derivative parts of the dual number
        val_access = loma_ir.StructAccess(loma_ir.Var(var_name), "val", t=loma_ir.Float())
        dval_access = loma_ir.StructAccess(loma_ir.Var(var_name), "dval", t=loma_ir.Float())
        input_vars.append(val_access)
        input_dvals.append(dval_access)
    
    # Process each layer
    prev_layer_outputs = input_vars
    prev_layer_doutputs = input_dvals
    
    for l in range(len(layer_sizes) - 1):
        current_layer_size = layer_sizes[l]
        next_layer_size = layer_sizes[l+1]
        
        layer_outputs = []
        layer_doutputs = []
        
        for i in range(next_layer_size):
            # Compute linear combination s_i^(l) and its derivative ds_i^(l)
            s_var_name = f"s_{l}_{i}"
            ds_var_name = f"ds_{l}_{i}"
            
            body.append(loma_ir.Declare(s_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            body.append(loma_ir.Declare(ds_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            
            s_var = loma_ir.Var(s_var_name, t=loma_ir.Float())
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
                
                body.append(loma_ir.Assign(
                    s_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        s_var,
                        weight_term,
                        t=loma_ir.Float()
                    )
                ))
                
                # ds_i += weight * dx_p
                dweight_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    weight_const,
                    prev_layer_doutputs[p],
                    t=loma_ir.Float()
                )
                
                body.append(loma_ir.Assign(
                    ds_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        ds_var,
                        dweight_term,
                        t=loma_ir.Float()
                    )
                ))
            
            # Apply nonlinearities and combine with alpha weights
            y_var_name = f"y_{l}_{i}"
            dy_var_name = f"dy_{l}_{i}"
            
            body.append(loma_ir.Declare(y_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            body.append(loma_ir.Declare(dy_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            
            y_var = loma_ir.Var(y_var_name, t=loma_ir.Float())
            dy_var = loma_ir.Var(dy_var_name, t=loma_ir.Float())
            
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
                
                # Get the derivative of the nonlinearity
                derivative_var_name = f"{nonlinearity_type}_deriv_{l}_{i}_{q}"
                derivative_output = kan_utils.apply_nonlinearity_derivative(
                    nonlinearity_type, 
                    s_var, 
                    body, 
                    derivative_var_name
                )
                
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
                
                # dy_i += alpha * derivative * ds_i
                deriv_ds_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    derivative_output,
                    ds_var,
                    t=loma_ir.Float()
                )
                
                weighted_deriv_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_const,
                    deriv_ds_term,
                    t=loma_ir.Float()
                )
                
                body.append(loma_ir.Assign(
                    dy_var,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        dy_var,
                        weighted_deriv_term,
                        t=loma_ir.Float()
                    )
                ))
            
            layer_outputs.append(y_var)
            layer_doutputs.append(dy_var)
        
        prev_layer_outputs = layer_outputs
        prev_layer_doutputs = layer_doutputs
    
    # Create the return value (dual number containing value and derivative)
    if output_size == 1:
        # Create a _dfloat struct with the value and derivative
        ret_var_name = "ret"
        body.append(loma_ir.Declare(ret_var_name, loma_ir.Struct("_dfloat", []), None))
        ret_var = loma_ir.Var(ret_var_name, t=loma_ir.Struct("_dfloat", []))
        
        # Set the value part
        body.append(loma_ir.Assign(
            loma_ir.StructAccess(ret_var, "val", t=loma_ir.Float()),
            prev_layer_outputs[0]
        ))
        
        # Set the derivative part
        body.append(loma_ir.Assign(
            loma_ir.StructAccess(ret_var, "dval", t=loma_ir.Float()),
            prev_layer_doutputs[0]
        ))
        
        body.append(loma_ir.Return(ret_var))
    else:
        # TODO: Handle multi-output case by creating appropriate struct
        pass
    
    return loma_ir.FunctionDef(
        diff_func_id,
        args,
        body,
        False,  # is_simd
        loma_ir.Struct("_dfloat", []) if output_size == 1 else None,  # ret_type
    )


def generate_random_string(length=6):
    """Generate a random string of fixed length"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def create_kan_forward_diff(diff_func_id,
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
    Create a forward differentiation function for a KAN network to be used with autodiff
    
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
        A Loma IR function definition for the KAN forward differentiation
    """
    if nonlinearities is None:
        # Default: use all 6 nonlinearities
        nonlinearities = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softplus', 'elu']
        
    if weights is None:
        weights = {}
        
    if alpha_weights is None:
        alpha_weights = {}
    
    # Ensure _dfloat struct exists
    if '_dfloat' not in structs:
        dfloat_struct = loma_ir.Struct(
            '_dfloat',
            [
                loma_ir.MemberDef('val', loma_ir.Float()),
                loma_ir.MemberDef('dval', loma_ir.Float())
            ],
            None
        )
        structs['_dfloat'] = dfloat_struct
        diff_structs[loma_ir.Float()] = dfloat_struct
    
    # Create a function to make _dfloat instances
    if 'make__dfloat' not in funcs:
        make_dfloat_args = [
            loma_ir.Arg('val', loma_ir.Float(), loma_ir.In()),
            loma_ir.Arg('dval', loma_ir.Float(), loma_ir.In())
        ]
        
        make_dfloat_body = []
        ret_var_name = 'ret'
        make_dfloat_body.append(loma_ir.Declare(ret_var_name, loma_ir.Struct('_dfloat', []), None))
        ret_var = loma_ir.Var(ret_var_name, t=loma_ir.Struct('_dfloat', []))
        
        make_dfloat_body.append(loma_ir.Assign(
            loma_ir.StructAccess(ret_var, 'val', t=loma_ir.Float()),
            loma_ir.Var('val', t=loma_ir.Float())
        ))
        
        make_dfloat_body.append(loma_ir.Assign(
            loma_ir.StructAccess(ret_var, 'dval', t=loma_ir.Float()),
            loma_ir.Var('dval', t=loma_ir.Float())
        ))
        
        make_dfloat_body.append(loma_ir.Return(ret_var))
        
        make_dfloat_func = loma_ir.FunctionDef(
            'make__dfloat',
            make_dfloat_args,
            make_dfloat_body,
            False,
            loma_ir.Struct('_dfloat', [])
        )
        
        funcs['make__dfloat'] = make_dfloat_func
    
    # Create the forward differentiation function
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