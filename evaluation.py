import sys
import os
import ast
import operator as op

def eval_expr(expr):
    node = ast.parse(expr, mode='eval')
    return eval_equation(node.body)

def eval_equation(node):
    '''
    Evaluate the Expr
    '''
    operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
                ast.USub: op.neg}
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return operators[type(node.op)](eval_equation(node.left), eval_equation(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_equation(node.operand))
    else:
        raise TypeError(node)
