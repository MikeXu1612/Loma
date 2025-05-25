import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import math
import numpy as np

# Activation functions and their derivatives as defined in the KAN paper

def sigmoid(x):
    """Sigmoid activation function: φ(s) = 1/(1+e^-s)"""
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid: φ'(s) = φ(s)(1 - φ(s))"""
    sig = sigmoid(x)
    return sig * (1.0 - sig)

def tanh(x):
    """Tanh activation function: φ(s) = tanh(s)"""
    return math.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh: φ'(s) = 1 - tanh^2(s)"""
    t = math.tanh(x)
    return 1.0 - t * t

def relu(x):
    """ReLU activation function: φ(s) = max(0, s)"""
    return max(0.0, x)

def relu_derivative(x):
    """Derivative of ReLU: φ'(s) = {1 if s > 0, 0 if s ≤ 0}"""
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function: φ(s) = max(αs, s)"""
    return max(alpha * x, x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU: φ'(s) = {1 if s > 0, α if s ≤ 0}"""
    return 1.0 if x > 0 else alpha

def softplus(x):
    """Softplus activation function: φ(s) = ln(1 + e^s)"""
    return math.log(1.0 + math.exp(x))

def softplus_derivative(x):
    """Derivative of Softplus: φ'(s) = 1/(1+e^-s)"""
    return 1.0 / (1.0 + math.exp(-x))

def elu(x, alpha=1.0):
    """ELU activation function: φ(s) = {s if s > 0, α(e^s - 1) if s ≤ 0}"""
    return x if x > 0 else alpha * (math.exp(x) - 1.0)

def elu_derivative(x, alpha=1.0):
    """Derivative of ELU: φ'(s) = {1 if s > 0, αe^s if s ≤ 0}"""
    return 1.0 if x > 0 else alpha * math.exp(x)

# Maps for activation functions and their derivatives
ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'softplus': softplus,
    'elu': elu
}

ACTIVATION_DERIVATIVES = {
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'relu': relu_derivative,
    'leaky_relu': leaky_relu_derivative,
    'softplus': softplus_derivative,
    'elu': elu_derivative
}

# Functions to create Loma IR nodes for activations and their derivatives

def create_sigmoid_ir(s_var):
    """Create Loma IR for sigmoid activation function"""
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
    
    return loma_ir.BinaryOp(
        loma_ir.Div(),
        loma_ir.ConstFloat(1.0),
        denominator,
        t=loma_ir.Float()
    )

def create_sigmoid_derivative_ir(s_var, sigmoid_output=None):
    """Create Loma IR for sigmoid derivative"""
    if sigmoid_output is None:
        sigmoid_output = create_sigmoid_ir(s_var)
    
    one_minus_sigmoid = loma_ir.BinaryOp(
        loma_ir.Sub(),
        loma_ir.ConstFloat(1.0),
        sigmoid_output,
        t=loma_ir.Float()
    )
    
    return loma_ir.BinaryOp(
        loma_ir.Mul(),
        sigmoid_output,
        one_minus_sigmoid,
        t=loma_ir.Float()
    )

def create_tanh_ir(s_var):
    """Create Loma IR for tanh activation function"""
    return loma_ir.Call(
        "tanh",
        [s_var],
        t=loma_ir.Float()
    )

def create_tanh_derivative_ir(s_var, tanh_output=None):
    """Create Loma IR for tanh derivative"""
    if tanh_output is None:
        tanh_output = create_tanh_ir(s_var)
    
    tanh_squared = loma_ir.BinaryOp(
        loma_ir.Mul(),
        tanh_output,
        tanh_output,
        t=loma_ir.Float()
    )
    
    return loma_ir.BinaryOp(
        loma_ir.Sub(),
        loma_ir.ConstFloat(1.0),
        tanh_squared,
        t=loma_ir.Float()
    )

def create_relu_ir(s_var, body=None, var_name=None):
    """Create Loma IR for ReLU activation function"""
    if body is None or var_name is None:
        # Just return the expression, no variable declaration
        return loma_ir.Call(
            "max",
            [loma_ir.ConstFloat(0.0), s_var],
            t=loma_ir.Float()
        )
    
    # If body is provided, add statements for if-else implementation
    zero = loma_ir.ConstFloat(0.0)
    nonlinear_var = loma_ir.Var(var_name, t=loma_ir.Float())
    
    body.append(loma_ir.Declare(var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
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
    
    return nonlinear_var

def create_relu_derivative_ir(s_var, body=None, var_name=None):
    """Create Loma IR for ReLU derivative"""
    if body is None or var_name is None:
        # Just return the conditional expression
        return loma_ir.BinaryOp(
            loma_ir.Greater(),
            s_var,
            loma_ir.ConstFloat(0.0),
            t=loma_ir.Float()
        )
    
    # If body is provided, add statements for if-else implementation
    nonlinear_var = loma_ir.Var(var_name, t=loma_ir.Float())
    
    body.append(loma_ir.Declare(var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
    body.append(loma_ir.IfElse(
        loma_ir.BinaryOp(
            loma_ir.Greater(),
            s_var,
            loma_ir.ConstFloat(0.0),
            t=loma_ir.Float()
        ),
        [loma_ir.Assign(nonlinear_var, loma_ir.ConstFloat(1.0))],
        [loma_ir.Assign(nonlinear_var, loma_ir.ConstFloat(0.0))]
    ))
    
    return nonlinear_var

def create_leaky_relu_ir(s_var, body=None, var_name=None, alpha=0.01):
    """Create Loma IR for Leaky ReLU activation function"""
    alpha_const = loma_ir.ConstFloat(alpha)
    alpha_s = loma_ir.BinaryOp(
        loma_ir.Mul(),
        alpha_const,
        s_var,
        t=loma_ir.Float()
    )
    
    if body is None or var_name is None:
        # Just return the expression
        return loma_ir.Call(
            "max",
            [alpha_s, s_var],
            t=loma_ir.Float()
        )
    
    # If body is provided, add statements for if-else implementation
    nonlinear_var = loma_ir.Var(var_name, t=loma_ir.Float())
    
    body.append(loma_ir.Declare(var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
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
    
    return nonlinear_var

def create_leaky_relu_derivative_ir(s_var, body=None, var_name=None, alpha=0.01):
    """Create Loma IR for Leaky ReLU derivative"""
    alpha_const = loma_ir.ConstFloat(alpha)
    
    if body is None or var_name is None:
        # Just return a conditional expression
        return loma_ir.BinaryOp(
            loma_ir.Greater(),
            s_var,
            loma_ir.ConstFloat(0.0),
            t=loma_ir.Float()
        )
    
    # If body is provided, add statements for if-else implementation
    nonlinear_var = loma_ir.Var(var_name, t=loma_ir.Float())
    
    body.append(loma_ir.Declare(var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
    body.append(loma_ir.IfElse(
        loma_ir.BinaryOp(
            loma_ir.Greater(),
            s_var,
            loma_ir.ConstFloat(0.0),
            t=loma_ir.Float()
        ),
        [loma_ir.Assign(nonlinear_var, loma_ir.ConstFloat(1.0))],
        [loma_ir.Assign(nonlinear_var, alpha_const)]
    ))
    
    return nonlinear_var

def create_softplus_ir(s_var):
    """Create Loma IR for Softplus activation function"""
    exp_term = loma_ir.Call(
        "exp",
        [s_var],
        t=loma_ir.Float()
    )
    
    return loma_ir.Call(
        "log",
        [loma_ir.BinaryOp(
            loma_ir.Add(),
            loma_ir.ConstFloat(1.0),
            exp_term,
            t=loma_ir.Float()
        )],
        t=loma_ir.Float()
    )

def create_softplus_derivative_ir(s_var):
    """Create Loma IR for Softplus derivative (which is sigmoid)"""
    return create_sigmoid_ir(s_var)

def create_elu_ir(s_var, body=None, var_name=None, alpha=1.0):
    """Create Loma IR for ELU activation function"""
    alpha_const = loma_ir.ConstFloat(alpha)
    exp_s = loma_ir.Call(
        "exp",
        [s_var],
        t=loma_ir.Float()
    )
    
    alpha_exp_minus_1 = loma_ir.BinaryOp(
        loma_ir.Mul(),
        alpha_const,
        loma_ir.BinaryOp(
            loma_ir.Sub(),
            exp_s,
            loma_ir.ConstFloat(1.0),
            t=loma_ir.Float()
        ),
        t=loma_ir.Float()
    )
    
    if body is None or var_name is None:
        # Just return a conditional expression
        return loma_ir.BinaryOp(
            loma_ir.Greater(),
            s_var,
            loma_ir.ConstFloat(0.0),
            t=loma_ir.Float()
        )
    
    # If body is provided, add statements for if-else implementation
    nonlinear_var = loma_ir.Var(var_name, t=loma_ir.Float())
    
    body.append(loma_ir.Declare(var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
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
    
    return nonlinear_var

def create_elu_derivative_ir(s_var, body=None, var_name=None, alpha=1.0):
    """Create Loma IR for ELU derivative"""
    alpha_const = loma_ir.ConstFloat(alpha)
    alpha_exp_s = loma_ir.BinaryOp(
        loma_ir.Mul(),
        alpha_const,
        loma_ir.Call(
            "exp",
            [s_var],
            t=loma_ir.Float()
        ),
        t=loma_ir.Float()
    )
    
    if body is None or var_name is None:
        # Just return a conditional expression
        return loma_ir.BinaryOp(
            loma_ir.Greater(),
            s_var,
            loma_ir.ConstFloat(0.0),
            t=loma_ir.Float()
        )
    
    # If body is provided, add statements for if-else implementation
    nonlinear_var = loma_ir.Var(var_name, t=loma_ir.Float())
    
    body.append(loma_ir.Declare(var_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
    body.append(loma_ir.IfElse(
        loma_ir.BinaryOp(
            loma_ir.Greater(),
            s_var,
            loma_ir.ConstFloat(0.0),
            t=loma_ir.Float()
        ),
        [loma_ir.Assign(nonlinear_var, loma_ir.ConstFloat(1.0))],
        [loma_ir.Assign(nonlinear_var, alpha_exp_s)]
    ))
    
    return nonlinear_var

# Map of activation function IR creators
ACTIVATION_IR_CREATORS = {
    'sigmoid': create_sigmoid_ir,
    'tanh': create_tanh_ir,
    'relu': create_relu_ir,
    'leaky_relu': create_leaky_relu_ir,
    'softplus': create_softplus_ir,
    'elu': create_elu_ir
}

# Map of activation derivative IR creators
ACTIVATION_DERIVATIVE_IR_CREATORS = {
    'sigmoid': create_sigmoid_derivative_ir,
    'tanh': create_tanh_derivative_ir,
    'relu': create_relu_derivative_ir,
    'leaky_relu': create_leaky_relu_derivative_ir,
    'softplus': create_softplus_derivative_ir,
    'elu': create_elu_derivative_ir
}

def apply_nonlinearity(nonlinearity_type, s_var, body=None, var_name=None):
    """Apply a specific nonlinearity to s_var in Loma IR"""
    if nonlinearity_type in ACTIVATION_IR_CREATORS:
        return ACTIVATION_IR_CREATORS[nonlinearity_type](s_var, body, var_name)
    else:
        raise ValueError(f"Unsupported nonlinearity type: {nonlinearity_type}")

def apply_nonlinearity_derivative(nonlinearity_type, s_var, body=None, var_name=None):
    """Apply the derivative of a specific nonlinearity to s_var in Loma IR"""
    if nonlinearity_type in ACTIVATION_DERIVATIVE_IR_CREATORS:
        return ACTIVATION_DERIVATIVE_IR_CREATORS[nonlinearity_type](s_var, body, var_name)
    else:
        raise ValueError(f"Unsupported nonlinearity type: {nonlinearity_type}") 