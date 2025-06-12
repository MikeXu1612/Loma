import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
from . import kan_utils
import random
import string
"""
def build_kan_layer_forward_ir(layer_idx, input_vals, input_dvals, current_layer_size, next_layer_size, nonlinearities, weights, alpha_weights, body):
    layer_outputs = []
    layer_doutputs = []

    for i in range(next_layer_size):
        s_name = f"s_{layer_idx}_{i}"
        ds_name = f"ds_{layer_idx}_{i}"
        body.append(loma_ir.Declare(s_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
        body.append(loma_ir.Declare(ds_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
        s_var = loma_ir.Var(s_name, t=loma_ir.Float())
        ds_var = loma_ir.Var(ds_name, t=loma_ir.Float())

        for p in range(current_layer_size):
            weight = weights.get(f"w_{layer_idx}_{i}_{p}", random.uniform(-0.1, 0.1))
            weight_const = loma_ir.ConstFloat(weight)
            
            body.append(loma_ir.Assign(
                s_var,
                loma_ir.BinaryOp(loma_ir.Add(), s_var, loma_ir.BinaryOp(loma_ir.Mul(), weight_const, input_vals[p], t=loma_ir.Float()), t=loma_ir.Float())
            ))

            body.append(loma_ir.Assign(
                ds_var,
                loma_ir.BinaryOp(loma_ir.Add(), ds_var, loma_ir.BinaryOp(loma_ir.Mul(), weight_const, input_dvals[p], t=loma_ir.Float()), t=loma_ir.Float())
            ))

        y_name = f"y_{layer_idx}_{i}"
        dy_name = f"dy_{layer_idx}_{i}"
        body.append(loma_ir.Declare(y_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
        body.append(loma_ir.Declare(dy_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
        y_var = loma_ir.Var(y_name, t=loma_ir.Float())
        dy_var = loma_ir.Var(dy_name, t=loma_ir.Float())

        for q, nonlinearity_type in enumerate(nonlinearities):
            alpha = alpha_weights.get(f"alpha_{layer_idx}_{i}_{q}", random.uniform(0, 1))
            alpha_const = loma_ir.ConstFloat(alpha)

            phi_var_name = f"phi_{layer_idx}_{i}_{q}"
            dphi_var_name = f"dphi_{layer_idx}_{i}_{q}"

            phi = kan_utils.apply_nonlinearity(nonlinearity_type, s_var, body, phi_var_name)
            dphi = kan_utils.apply_nonlinearity_derivative(nonlinearity_type, s_var, body, dphi_var_name)

            body.append(loma_ir.Assign(
                y_var,
                loma_ir.BinaryOp(loma_ir.Add(), y_var, loma_ir.BinaryOp(loma_ir.Mul(), alpha_const, phi, t=loma_ir.Float()), t=loma_ir.Float())
            ))

            body.append(loma_ir.Assign(
                dy_var,
                loma_ir.BinaryOp(
                    loma_ir.Add(),
                    dy_var,
                    loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        alpha_const,
                        loma_ir.BinaryOp(loma_ir.Mul(), dphi, ds_var, t=loma_ir.Float()),
                        t=loma_ir.Float()
                    ),
                    t=loma_ir.Float()
                )
            ))

        layer_outputs.append(y_var)
        layer_doutputs.append(dy_var)

    return layer_outputs, layer_doutputs

def kan_forward_diff_pass(diff_func_id, func_id, input_size, output_size, hidden_sizes, nonlinearities, weights, alpha_weights):
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    args = [loma_ir.Arg(f"x{i}", loma_ir.Struct("_dfloat", []), loma_ir.In()) for i in range(input_size)]
    body = []

    input_vals = [loma_ir.StructAccess(loma_ir.Var(f"x{i}"), "val", t=loma_ir.Float()) for i in range(input_size)]
    input_dvals = [loma_ir.StructAccess(loma_ir.Var(f"x{i}"), "dval", t=loma_ir.Float()) for i in range(input_size)]

    prev_vals, prev_dvals = input_vals, input_dvals

    for l in range(len(layer_sizes) - 1):
        prev_vals, prev_dvals = build_kan_layer_forward_ir(
            l, prev_vals, prev_dvals,
            layer_sizes[l], layer_sizes[l+1],
            nonlinearities, weights, alpha_weights, body
        )

    if output_size == 1:
        ret = loma_ir.Var("ret", t=loma_ir.Struct("_dfloat", []))
        body.append(loma_ir.Declare(ret.id, ret.t, None))
        body.append(loma_ir.Assign(loma_ir.StructAccess(ret, "val", t=loma_ir.Float()), prev_vals[0]))
        body.append(loma_ir.Assign(loma_ir.StructAccess(ret, "dval", t=loma_ir.Float()), prev_dvals[0]))
        body.append(loma_ir.Return(ret))
        ret_type = loma_ir.Struct("_dfloat", [])
    else:
        result_type = loma_ir.Array(loma_ir.Struct("_dfloat", []), static_size=output_size)
        result_var = loma_ir.Var("result", t=result_type)
        body.append(loma_ir.Declare("result", result_type))
        for i in range(output_size):
            access = loma_ir.ArrayAccess(result_var, loma_ir.ConstInt(i), t=loma_ir.Struct("_dfloat", []))
            body.append(loma_ir.Assign(loma_ir.StructAccess(access, "val", t=loma_ir.Float()), prev_vals[i]))
            body.append(loma_ir.Assign(loma_ir.StructAccess(access, "dval", t=loma_ir.Float()), prev_dvals[i]))
        body.append(loma_ir.Return(result_var))
        ret_type = result_type

    return loma_ir.FunctionDef(diff_func_id, args, body, False, ret_type)

def generate_random_string(length=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def create_kan_forward_diff(diff_func_id, structs, funcs, diff_structs, func_id, input_size, output_size, hidden_sizes=[10], nonlinearities=None, weights=None, alpha_weights=None):
    if nonlinearities is None:
        nonlinearities = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softplus', 'elu']
    if weights is None:
        weights = {}
    if alpha_weights is None:
        alpha_weights = {}

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

    if 'make__dfloat' not in funcs:
        make_dfloat_args = [
            loma_ir.Arg('val', loma_ir.Float(), loma_ir.In()),
            loma_ir.Arg('dval', loma_ir.Float(), loma_ir.In())
        ]
        make_dfloat_body = []
        ret = loma_ir.Var('ret', t=loma_ir.Struct('_dfloat', []))
        make_dfloat_body.append(loma_ir.Declare('ret', ret.t, None))
        make_dfloat_body.append(loma_ir.Assign(loma_ir.StructAccess(ret, 'val', t=loma_ir.Float()), loma_ir.Var('val', t=loma_ir.Float())))
        make_dfloat_body.append(loma_ir.Assign(loma_ir.StructAccess(ret, 'dval', t=loma_ir.Float()), loma_ir.Var('dval', t=loma_ir.Float())))
        make_dfloat_body.append(loma_ir.Return(ret))
        funcs['make__dfloat'] = loma_ir.FunctionDef('make__dfloat', make_dfloat_args, make_dfloat_body, False, loma_ir.Struct('_dfloat', []))

    return kan_forward_diff_pass(
        diff_func_id, func_id, input_size, output_size, hidden_sizes, nonlinearities, weights, alpha_weights
    )
"""
STRUCT_CACHE = {}
import hashlib

def build_param_kan_layer_forward_ir(layer_idx, prev_vals, prev_dvals, current_layer_size, next_layer_size, nonlinearities, W, A, weight_offset, alpha_offset, body):
    layer_outputs = []
    layer_doutputs = []

    for i in range(next_layer_size):
        s_name = f"s_{layer_idx}_{i}"
        ds_name = f"ds_{layer_idx}_{i}"
        tmp_s_name = f"tmp_s_{layer_idx}_{i}"
        tmp_ds_name = f"tmp_ds_{layer_idx}_{i}"

        body.append(loma_ir.Declare(tmp_s_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
        body.append(loma_ir.Declare(tmp_ds_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
        tmp_s_var = loma_ir.Var(tmp_s_name, t=loma_ir.Float())
        tmp_ds_var = loma_ir.Var(tmp_ds_name, t=loma_ir.Float())

        for p in range(current_layer_size):
            w_idx = loma_ir.ConstInt(weight_offset + i * current_layer_size + p)
            w_val = loma_ir.ArrayAccess(W, w_idx, t=loma_ir.Float())
            body.append(loma_ir.Assign(
                tmp_s_var,
                loma_ir.BinaryOp(loma_ir.Add(), tmp_s_var, loma_ir.BinaryOp(loma_ir.Mul(), w_val, prev_vals[p], t=loma_ir.Float()), t=loma_ir.Float())
            ))
            body.append(loma_ir.Assign(
                tmp_ds_var,
                loma_ir.BinaryOp(loma_ir.Add(), tmp_ds_var, loma_ir.BinaryOp(loma_ir.Mul(), w_val, prev_dvals[p], t=loma_ir.Float()), t=loma_ir.Float())
            ))

        body.append(loma_ir.Declare(s_name, loma_ir.Float(), tmp_s_var))
        body.append(loma_ir.Declare(ds_name, loma_ir.Float(), tmp_ds_var))
        s_var = loma_ir.Var(s_name, t=loma_ir.Float())
        ds_var = loma_ir.Var(ds_name, t=loma_ir.Float())

        y_name = f"y_{layer_idx}_{i}"
        dy_name = f"dy_{layer_idx}_{i}"
        body.append(loma_ir.Declare(y_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
        body.append(loma_ir.Declare(dy_name, loma_ir.Float(), loma_ir.ConstFloat(0.0)))
        y_var = loma_ir.Var(y_name, t=loma_ir.Float())
        dy_var = loma_ir.Var(dy_name, t=loma_ir.Float())

        num_nl = len(nonlinearities)
        for q, nl_type in enumerate(nonlinearities):
            a_idx = loma_ir.ConstInt(alpha_offset + i * num_nl + q)
            a_val = loma_ir.ArrayAccess(A, a_idx, t=loma_ir.Float())
            phi = kan_utils.apply_nonlinearity(nl_type, s_var, body, f"phi_{layer_idx}_{i}_{q}")
            dphi = kan_utils.apply_nonlinearity_derivative(nl_type, s_var, body, f"dphi_{layer_idx}_{i}_{q}")
            body.append(loma_ir.Assign(
                y_var,
                loma_ir.BinaryOp(loma_ir.Add(), y_var, loma_ir.BinaryOp(loma_ir.Mul(), a_val, phi, t=loma_ir.Float()), t=loma_ir.Float())
            ))
            term = loma_ir.BinaryOp(loma_ir.Mul(), dphi, ds_var, t=loma_ir.Float())
            body.append(loma_ir.Assign(
                dy_var,
                loma_ir.BinaryOp(loma_ir.Add(), dy_var, loma_ir.BinaryOp(loma_ir.Mul(), a_val, term, t=loma_ir.Float()), t=loma_ir.Float())
            ))

        layer_outputs.append(y_var)
        layer_doutputs.append(dy_var)

    return layer_outputs, layer_doutputs


def param_kan_forward_diff(diff_func_id, input_size, output_size, hidden_sizes, nonlinearities):
    num_nl = len(nonlinearities)
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    weight_len = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
    alpha_len = sum(layer_sizes[i+1] * num_nl for i in range(len(layer_sizes)-1))

    args = [
        loma_ir.Arg("X", loma_ir.Array(loma_ir.Struct("_dfloat", []), static_size=input_size), loma_ir.In()),
        loma_ir.Arg("W", loma_ir.Array(loma_ir.Float(), static_size=weight_len), loma_ir.In()),
        loma_ir.Arg("A", loma_ir.Array(loma_ir.Float(), static_size=alpha_len), loma_ir.In()),
        loma_ir.Arg("Y", loma_ir.Array(loma_ir.Struct("_dfloat", []), static_size=output_size), loma_ir.Out())
    ]
    body = []

    X = loma_ir.Var("X")
    prev_vals = [loma_ir.StructAccess(loma_ir.ArrayAccess(X, loma_ir.ConstInt(i), t=loma_ir.Struct("_dfloat", [])), "val", t=loma_ir.Float()) for i in range(input_size)]
    prev_dvals = [loma_ir.StructAccess(loma_ir.ArrayAccess(X, loma_ir.ConstInt(i), t=loma_ir.Struct("_dfloat", [])), "dval", t=loma_ir.Float()) for i in range(input_size)]

    weight_offset = 0
    alpha_offset = 0
    for l in range(len(layer_sizes)-1):
        cur_size = layer_sizes[l]
        nxt_size = layer_sizes[l+1]
        prev_vals, prev_dvals = build_param_kan_layer_forward_ir(
            l, prev_vals, prev_dvals, cur_size, nxt_size,
            nonlinearities, loma_ir.Var("W"), loma_ir.Var("A"),
            weight_offset, alpha_offset, body
        )
        weight_offset += cur_size * nxt_size
        alpha_offset += nxt_size * num_nl

    Y = loma_ir.Var("Y")
    for i in range(output_size):
        cell = loma_ir.ArrayAccess(Y, loma_ir.ConstInt(i), t=loma_ir.Struct("_dfloat", []))
        body.append(loma_ir.Assign(loma_ir.StructAccess(cell, "val", t=loma_ir.Float()), prev_vals[i]))
        body.append(loma_ir.Assign(loma_ir.StructAccess(cell, "dval", t=loma_ir.Float()), prev_dvals[i]))

    return loma_ir.FunctionDef(diff_func_id, args, body, False, None)

def create_param_kan_forward_diff(diff_func_id, structs, funcs, diff_structs, input_size, output_size, hidden_sizes, nonlinearities=None):
    if nonlinearities is None:
        nonlinearities = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'softplus', 'elu']

    key = (tuple(hidden_sizes), input_size, output_size, tuple(nonlinearities))
    hash_key = hashlib.sha1(str(key).encode()).hexdigest()[:8]
    template_name = f"d_kan_layer_{hash_key}"

    if '_dfloat' not in structs:
        dfloat = loma_ir.Struct('_dfloat', [loma_ir.MemberDef('val', loma_ir.Float()), loma_ir.MemberDef('dval', loma_ir.Float())], None)
        structs['_dfloat'] = dfloat
        diff_structs[loma_ir.Float()] = dfloat

    if key not in STRUCT_CACHE:
        STRUCT_CACHE[key] = param_kan_forward_diff(template_name, input_size, output_size, hidden_sizes, nonlinearities)
        funcs[template_name] = STRUCT_CACHE[key]

    template = STRUCT_CACHE[key]
    call = loma_ir.Call(template.id, [loma_ir.Var(arg.id, t=arg.t) for arg in template.args], t=None)
    func_def = loma_ir.FunctionDef(
        id=diff_func_id,
        args=[loma_ir.Arg(arg.id, arg.t, arg.i) for arg in template.args],
        body=list(template.body),
        is_simd=False,
        ret_type=None
    )
    funcs[diff_func_id] = func_def
    return func_def
