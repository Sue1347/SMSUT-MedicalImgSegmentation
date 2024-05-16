# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn


def traverse_graph(var):
    """
    Args:
        var: output Variable
    """

    seen = set()
    var_lst = []

    def add_nodes(var):
        if var not in seen:
            if hasattr(var, 'variable'):
                u = var.variable
                if isinstance(u, nn.Parameter):
                    var_lst.append(u)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(u[0])

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    return var_lst


def make_closure(loss, net):
    def closure():
        used_vars = traverse_graph(loss)
        loss.backward()

        for p in net.parameters():
            exists = False
            for v in used_vars:
                exists = (p is v)
                if exists:
                    break
            if not exists:
                p.grad = None

        return loss

    return closure


def make_closure_fast(loss, net):
    def closure():
        used_vars = set(traverse_graph(loss))
        loss.backward()

        for p in net.parameters():
            if p not in used_vars:
                p.grad = None

        return loss

    return closure
