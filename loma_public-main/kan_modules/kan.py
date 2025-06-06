import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import autodiff
import numpy as np
import random

def parse_config_dict(dict_node):
    """解析 loma_ir.Dict 为 Python dict"""
    assert isinstance(dict_node, loma_ir.Dict)
    config = {}
    for k, v in zip(dict_node.keys, dict_node.values):
        assert isinstance(k, loma_ir.ConstStr)
        key = k.val
        if isinstance(v, loma_ir.ConstInt):
            config[key] = v.val
        elif isinstance(v, loma_ir.ConstFloat):
            config[key] = v.val
        elif isinstance(v, loma_ir.ArrayLiteral):
            config[key] = [e.val for e in v.elements if isinstance(e, loma_ir.ConstInt)]
        else:
            raise ValueError(f"Unsupported config value type: {type(v)}")
    return config


class KANLayer:
    """
    Implementation of a Kolmogorov-Arnold Network layer.
    Each node in the layer uses multiple non-linear functions to compute its output.
    """
    def __init__(self, input_size, output_size, num_nonlinearities=5):
        self.input_size = input_size
        self.output_size = output_size
        self.num_nonlinearities = num_nonlinearities
        
        # Weight matrices for the linear combinations
        self.weights = None
        
        # Weights for combining nonlinearities
        self.alpha_weights = None
        
    def initialize_weights(self):
        """Initialize the weights for linear combinations and nonlinearity mixing"""
        # Initialize weights for linear combinations (a_pi^(l) in the paper)
        self.weights = np.random.randn(self.output_size, self.input_size) * 0.1
        
        # Initialize weights for nonlinearity mixing (alpha_i^(q,l) in the paper)
        self.alpha_weights = np.random.randn(self.output_size, self.num_nonlinearities) * 0.1
        
        # Normalize alpha weights to sum to 1
        self.alpha_weights = self.alpha_weights / np.sum(self.alpha_weights, axis=1, keepdims=True)


class KANNetwork:
    """
    Implementation of a Kolmogorov-Arnold Network (KAN).
    """
    def __init__(self, layer_sizes, num_nonlinearities=5):
        """
        Initialize a KAN with specified layer sizes
        
        Args:
            layer_sizes: List of integers specifying the size of each layer
            num_nonlinearities: Number of nonlinear functions to use at each node
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.num_nonlinearities = num_nonlinearities
        
        # Initialize layers
        self.layers = []
        for i in range(self.num_layers):
            layer = KANLayer(layer_sizes[i], layer_sizes[i+1], num_nonlinearities)
            layer.initialize_weights()
            self.layers.append(layer)
    
    def forward(self, x):
        """
        Forward pass through the KAN network
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after forward pass
        """
        activations = x
        
        for layer in self.layers:
            # Compute linear combinations (s_i^(l))
            linear_combinations = np.dot(layer.weights, activations)
            
            # Apply multiple nonlinear functions and combine them
            outputs = np.zeros(layer.output_size)
            
            for i in range(layer.output_size):
                s_i = linear_combinations[i]
                y_i = 0
                
                for q in range(layer.num_nonlinearities):
                    # Apply nonlinearity q to s_i
                    nonlinear_output = self.apply_nonlinearity(s_i, q)
                    
                    # Combine with alpha weight
                    y_i += layer.alpha_weights[i, q] * nonlinear_output
                
                outputs[i] = y_i
                
            activations = outputs
            
        return activations
    
    def apply_nonlinearity(self, x, nonlinearity_idx):
        """
        Apply a specific nonlinearity function
        
        Args:
            x: Input value
            nonlinearity_idx: Index of the nonlinearity to apply
            
        Returns:
            Result of applying the nonlinearity
        """
        # Map nonlinearity_idx to specific activation functions
        if nonlinearity_idx % 6 == 0:
            # Sigmoid: φ(s) = 1/(1+e^-s)
            return 1.0 / (1.0 + np.exp(-x))
        elif nonlinearity_idx % 6 == 1:
            # Tanh: φ(s) = tanh(s)
            return np.tanh(x)
        elif nonlinearity_idx % 6 == 2:
            # ReLU: φ(s) = max(0, s)
            return np.maximum(0, x)
        elif nonlinearity_idx % 6 == 3:
            # Leaky ReLU: φ(s) = max(αs, s) with α=0.01
            alpha = 0.01
            return np.maximum(alpha * x, x)
        elif nonlinearity_idx % 6 == 4:
            # Softplus: φ(s) = ln(1 + e^s)
            return np.log(1.0 + np.exp(x))
        else:
            # ELU: φ(s) = {s if s > 0; α(e^s - 1) if s ≤ 0} with α=1.0
            alpha = 1.0
            return x if x > 0 else alpha * (np.exp(x) - 1)


def create_kan_in_loma(func_id, input_size, output_size, hidden_sizes=[10], num_nonlinearities=5, weights=None, alphas=None):
    """
    Create a KAN network in Loma IR
    
    Args:
        func_id: The ID of the function to create
        input_size: Size of the input layer
        output_size: Size of the output layer
        hidden_sizes: List of hidden layer sizes
        num_nonlinearities: Number of nonlinear functions to use at each node
        
    Returns:
        A Loma IR function definition for the KAN
    """
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    # Create arguments for the function
    args = [
        loma_ir.Arg("X", loma_ir.Array(loma_ir.Float(), static_size=input_size), loma_ir.In())
    ]

    if isinstance(weights, loma_ir.Var) and weights.t is not None:
        args.append(loma_ir.Arg(weights.id, weights.t, loma_ir.In()))

    if isinstance(alphas, loma_ir.Var) and alphas.t is not None:
        args.append(loma_ir.Arg(alphas.id, alphas.t, loma_ir.In()))

    
    # Create body of the function
    body = []
    
    # Declare variables for inputs
    X = loma_ir.Var("X", t=loma_ir.Array(loma_ir.Float(), input_size))
    input_vars = [
        loma_ir.ArrayAccess(X, loma_ir.ConstInt(i), t=loma_ir.Float())
        for i in range(input_size)
    ]

    
    # Process each layer
    prev_layer_outputs = input_vars
    
    for l in range(len(layer_sizes) - 1):
        current_layer_size = layer_sizes[l]
        next_layer_size = layer_sizes[l+1]
        
        layer_outputs = []
        
        for i in range(next_layer_size):
            # Compute linear combination s_i^(l)
            s_var_name = f"s_{l}_{i}"
            body.append(loma_ir.Declare(s_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            s_var = loma_ir.Var(s_var_name, t=loma_ir.Float())
            

            for p in range(current_layer_size):

                if weights is not None:
                    if isinstance(weights, list):
                        weight_val = weights[l][i * current_layer_size + p]
                        weight_expr = loma_ir.ConstFloat(weight_val)
                    else:  # assume it's a Var (e.g., Var("W"))
                        weight_index = loma_ir.ConstInt(i * current_layer_size + p)
                        weight_expr = loma_ir.ArrayAccess(weights, weight_index, t=loma_ir.Float())
                else:
                    weight_expr = loma_ir.ConstFloat(random.uniform(-0.1, 0.1))


                weight_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    weight_expr, 
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

            # Apply nonlinearities and combine with alpha weights
            y_var_name = f"y_{l}_{i}"
            body.append(loma_ir.Declare(y_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
            y_var = loma_ir.Var(y_var_name, t=loma_ir.Float())
                
            if alphas is not None:
                if isinstance(alphas, list):
                    alpha_weights = alphas[l][i]
                else:
                    alpha_weights = []
                    for j in range(num_nonlinearities):
                        alpha_idx = loma_ir.ConstInt(i * num_nonlinearities + j)
                        alpha_weights.append(loma_ir.ArrayAccess(alphas, alpha_idx, t=loma_ir.Float()))
            else:
                alpha_weights = [random.uniform(0, 1) for _ in range(num_nonlinearities)]
                total = sum(alpha_weights)
                alpha_weights = [w / total for w in alpha_weights]
            
            for q in range(num_nonlinearities):
                # Apply nonlinearity q to s_i
                nonlinear_output = None
                
                if q % 6 == 0:
                    # Sigmoid: φ(s) = 1/(1+e^-s)
                    exp_term = loma_ir.Call(
                        "exp", 
                        [loma_ir.BinaryOp(
                            loma_ir.Sub(),
                            loma_ir.ConstFloat(0.0),
                            s_var,
                            t=loma_ir.Float()
                        )],
                        t=loma_ir.Float()
                    )
                    
                    denominator = loma_ir.BinaryOp(
                        loma_ir.Add(),
                        loma_ir.ConstFloat(1.0),
                        exp_term,
                        t=loma_ir.Float()
                    )
                    
                    nonlinear_output = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        loma_ir.ConstFloat(1.0),
                        denominator,
                        t=loma_ir.Float()
                    )
                    
                elif q % 6 == 1:
                    # Tanh: φ(s) = tanh(s)
                    nonlinear_output = loma_ir.Call(
                        "tanh",
                        [s_var],
                        t=loma_ir.Float()
                    )
                    
                elif q % 6 == 2:
                    # ReLU: φ(s) = max(0, s)
                    # Use conditional to implement max
                    zero = loma_ir.ConstFloat(0.0)
                    nonlinear_var_name = f"relu_{l}_{i}_{q}"
                    body.append(loma_ir.Declare(nonlinear_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
                    nonlinear_var = loma_ir.Var(nonlinear_var_name, t=loma_ir.Float())
                    
                    body.append(loma_ir.IfElse(
                        loma_ir.BinaryOp(
                            loma_ir.Greater(),
                            s_var,
                            zero,
                            t=loma_ir.Float()
                        ),
                        [loma_ir.Assign(nonlinear_var, s_var)],
                        [loma_ir.Assign(nonlinear_var, zero)]
                    ))
                    
                    nonlinear_output = nonlinear_var
                    
                elif q % 6 == 3:
                    # Leaky ReLU: φ(s) = max(αs, s) with α=0.01
                    alpha = loma_ir.ConstFloat(0.01)
                    alpha_s = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        alpha,
                        s_var,
                        t=loma_ir.Float()
                    )
                    
                    nonlinear_var_name = f"leaky_relu_{l}_{i}_{q}"
                    body.append(loma_ir.Declare(nonlinear_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
                    nonlinear_var = loma_ir.Var(nonlinear_var_name, t=loma_ir.Float())
                    
                    body.append(loma_ir.IfElse(
                        loma_ir.BinaryOp(
                            loma_ir.Greater(),
                            s_var,
                            loma_ir.ConstFloat(0.0),
                            t=loma_ir.Float()
                        ),
                        [loma_ir.Assign(nonlinear_var, s_var)],
                        [loma_ir.Assign(nonlinear_var, alpha_s)]
                    ))
                    
                    nonlinear_output = nonlinear_var
                    
                elif q % 6 == 4:
                    # Softplus: φ(s) = ln(1 + e^s)
                    exp_term = loma_ir.Call(
                        "exp",
                        [s_var],
                        t=loma_ir.Float()
                    )
                    
                    log_term = loma_ir.Call(
                        "log",
                        [loma_ir.BinaryOp(
                            loma_ir.Add(),
                            loma_ir.ConstFloat(1.0),
                            exp_term,
                            t=loma_ir.Float()
                        )],
                        t=loma_ir.Float()
                    )
                    
                    nonlinear_output = log_term
                    
                else:
                    # ELU: φ(s) = {s if s > 0; α(e^s - 1) if s ≤ 0} with α=1.0
                    alpha = loma_ir.ConstFloat(1.0)
                    exp_s = loma_ir.Call(
                        "exp",
                        [s_var],
                        t=loma_ir.Float()
                    )
                    
                    alpha_exp_minus_1 = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        alpha,
                        loma_ir.BinaryOp(
                            loma_ir.Sub(),
                            exp_s,
                            loma_ir.ConstFloat(1.0),
                            t=loma_ir.Float()
                        ),
                        t=loma_ir.Float()
                    )
                    
                    nonlinear_var_name = f"elu_{l}_{i}_{q}"
                    body.append(loma_ir.Declare(nonlinear_var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
                    nonlinear_var = loma_ir.Var(nonlinear_var_name, t=loma_ir.Float())
                    
                    body.append(loma_ir.IfElse(
                        loma_ir.BinaryOp(
                            loma_ir.Greater(),
                            s_var,
                            loma_ir.ConstFloat(0.0),
                            t=loma_ir.Float()
                        ),
                        [loma_ir.Assign(nonlinear_var, s_var)],
                        [loma_ir.Assign(nonlinear_var, alpha_exp_minus_1)]
                    ))
                    
                    nonlinear_output = nonlinear_var
                
                # Combine with alpha weight
                #alpha_weight = loma_ir.ConstFloat(alpha_weights[q])
                if isinstance(alpha_weights[q], loma_ir.expr):
                    alpha_weight = alpha_weights[q]
                else:
                    alpha_weight = loma_ir.ConstFloat(alpha_weights[q])


                weighted_term = loma_ir.BinaryOp(
                    loma_ir.Mul(),
                    alpha_weight,
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
            
            layer_outputs.append(y_var)
        
        prev_layer_outputs = layer_outputs
    
    # Return the final layer output
    if output_size == 1:
        body.append(loma_ir.Return(prev_layer_outputs[0]))
    else:
        # Create a struct to return multiple values
        # This would need proper struct definition in the actual implementation
        pass
    
    return loma_ir.FunctionDef(
        func_id,
        args,
        body,
        False,
        loma_ir.Float() if output_size == 1 else None,
    ) 

def make_kan_layer(name="kan_layer", input_size=2, output_size=1, hidden_sizes=[], num_nonlinearities=3, weights=None, alphas=None):
    return create_kan_in_loma(
        func_id=name,
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        num_nonlinearities=num_nonlinearities, 
        weights=weights,
        alphas=alphas
    )

def injectable_kan_layer():
    return make_kan_layer(name="kan_layer", input_size=2, output_size=1, hidden_sizes=[3], num_nonlinearities=6)
