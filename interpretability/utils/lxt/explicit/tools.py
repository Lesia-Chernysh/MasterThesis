from collections import namedtuple
from graphviz import Digraph
import torch
from torch.autograd import Variable
import warnings

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"

def get_fn_name(fn, show_attrs, max_attr_chars):
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = dict()
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX):]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width)+ 's'
    truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params


def traced_backward(var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50):
    """ Produces Graphviz representation of PyTorch autograd graph.

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:
     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires gradtensor) 
        show_attrs: whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved: whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
    """

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')
    dot = Digraph(node_attr=node_attr, graph_attr={'size':'12,12', 'splines': 'polyline'})
    seen = set()
    hooks = []

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return '%s\n %s' % (name, size_to_str(var.size()))

    def add_edge_on_backward(next_fn, fn, i=0):
        def hook(inpgrad, outgrad): 
            try:
                dot.edge(str(id(next_fn))+':s', str(id(fn)), f'{inpgrad[i].sum().item():.5f}')
            except:
                print('edge', next_fn, fn, i)
        # return hooks.append(fn.register_hook(lambda inpgrad, outgrad: dot.edge(str(id(next_fn))+':s', str(id(fn)), str(inpgrad[i].sum().item()))))
        return hooks.append(fn.register_hook(hook))

    def add_node_on_backward(fn, fn_name):
        def hook(grad):
            try:
                #xlabel = str(fn._forward_cls) if hasattr(fn, '_forward_cls') else None
                dot.node(str(id(fn)), label=fn_name + '\n' + f'{grad[0].sum().item():.5f}')#, xlabel=xlabel)
            except:
                print('node', fn, fn_name)
        #return hooks.append(fn.register_prehook(lambda grad: dot.node(str(id(fn)), label=fn_name + '\n' + str(grad[0].sum().item())))) #, xlabel=str(grad[0].sum().item()))))
        return hooks.append(fn.register_prehook(hook))
    
    def add_nodes(fn):
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        if hasattr(fn, 'variable'):
            # if grad_accumulator, add the node for `.variable`
            var = fn.variable
            seen.add(var)
            dot.node(str(id(var)), get_var_name(var), fillcolor='lightblue')
            dot.edge(str(id(var)), str(id(fn)))

        # add the node for this grad_fn
        add_node_on_backward(fn, get_fn_name(fn, show_attrs, max_attr_chars))

        # recurse
        if hasattr(fn, 'next_functions'):
            for i, u in enumerate(fn.next_functions):
                if u[0] is not None:
                    add_edge_on_backward(u[0], fn, i)
                    add_nodes(u[0])

        # note: this used to show .saved_tensors in pytorch0.2, but stopped
        # working* as it was moved to ATen and Variable-Tensor merged
        # also note that this still works for custom autograd functions
        # if hasattr(fn, 'saved_tensors'):
        #     for t in fn.saved_tensors:
        #         seen.add(t)
        #         dot.edge(str(id(t)), str(id(fn)), dir="none")
        #         dot.node(str(id(t)), get_var_name(t), fillcolor='orange')


    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if (var.grad_fn):
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)), str(1.0)) # if backward init with var --> var.item()
        if var._is_view():
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.edge(str(id(var._base)), str(id(var)), style="dotted")


    add_base_tensor(var)

    try:
        var.backward() # if backward init with var, change add_base_tensor
    finally:
        for hook in hooks:
            hook.remove()

    resize_graph(dot)

    return dot

def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)