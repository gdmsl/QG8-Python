#!/usr/bin/python3
# ops.py
# Ops library for QG8 python extension library.
#
# Author       : Shannon Whitlock <whitlock@unistra.fr>
# Date created : 18 July 2021
#

# Copyright 2021 University of Strasbourg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from qutip import *
import numpy as np
from qg8.constants import *
import functools

"""
This file contains instructions for processing qg8 op nodes called from the 
qg8.run() method. Each function takes a QG8Node object and processes them to
set the .outdata attribute depending on its assigned input nodes.

Arithmetic in this script is performed using the Numpy library and therefore
does not take fully advantage of the sparse format of qg8 tensors.

To define new types of nodes please define the corresponding operation as a
function and add it to the type registry with a unique integer id.
"""

def add(qg8_node):
    """
    element-wise addition of several nodes
    """
    qg8_node.outdata = sum([n.outdata for n in qg8_node.input_nodes])


def sum(qg8_node):
    data = [node.outdata for node in qg8_node.input_nodes]
    try:
        qg8_node.outdata = sum(data)
    except:
        print("Could not sum arrays at node: " + str(qg8_node))
        qg8_node.outdata = data[0]


def matmul(qg8_node):
    """
    External product between node outputs (dot product)

    Computes the product `a @ b @ c @ ...`
    """
    data = [node.outdata for node in qg8_node.input_nodes]
    functools.reduce(np.dot, data)


def multiply(qg8_node):
    """
    Element-wise multiplication between node outputs (Hadamand product)
    """
    data = [node.outdata for node in qg8_node.input_nodes]
    functools.reduce(np.multiply, data)


def join(qg8_node):
    qg8_node.outdata = [node.outdata for node in qg8_node.input_nodes]


def solvequtip(qg8_node):
    """
    Operator used to simulate the time dependent Schrödinger equation
    args:
        qg8_node: the current QG8 node

    qg8_node must link to a nested list of input nodes, structured as follows
    qg8_node.input_nodes = [psi0, times, [h1, c1], [h2, c2], ...] where,

        psi0: the initial state for the simulation (rank 1 tensor node)
        times:  time points for simulation results (rank 1 tensor node)
        h1, h2, ... : half-Hermitian operators (rank 2 tensor nodes)
        c1, c2, ... : time-dependent coefficients (rank 1 tensor nodes)

    Upon running the graph, this function updates qg8_node.outdata with a
    complex valued 2D matrix representing the quantum state at each time point
    (vertically stacked)
    """

    psi0 = Qobj(qg8_node.input_nodes[0].outdata)
    operators = [hamiltonian_term.outdata[0]
                 for hamiltonian_term in qg8_node.input_nodes[2::]]
    packing = [hamiltonian_term.input_nodes[0].tensor.packing
               for hamiltonian_term in qg8_node.input_nodes[2::]]
    coeffs = [hamiltonian_term.outdata[1]
              for hamiltonian_term in qg8_node.input_nodes[2::]]

    times = qg8_node.input_nodes[1].outdata.astype('float64')  # from the first track

    ops = []  # Build Qutip QobjEvo object

    for j, op in enumerate(operators):
        coeff = coeffs[j] * np.ones(len(times))  # convert to array in case constant
        ops.append([Qobj(op), coeff])

        if packing[j] == QG8_PACKING_HALFHERMITIAN:  # add h.c. terms
            ops.append([Qobj(np.conjugate(op.T)), np.conjugate(coeff)])

    H = QobjEvo(ops, tlist=times)
    options = Options(max_step=times[1] - times[0])

    result = sesolve(H, psi0, times, options=options).states
    qg8_node.outdata = np.vstack([psi.full().T for psi in result])


def expectationvalue(qg8_node):
    """
    returns a 1D tensor of expectation values
    """
    psi = qg8_node.input_nodes[0].outdata
    op = qg8_node.input_nodes[1].outdata

    expec = np.zeros(len(psi))
    for t,p in enumerate(psi):
        expec[t] = np.real(p.conjugate() @ op @ p.T)
    qg8_node.outdata = expec


def to_numpy(qg8_tensor):
    """
    Convert qg8_tensor to dense numpy array
    """
    dtype = dtype_to_name(qg8_tensor.dtype_id)
    ndarray = np.zeros(qg8_tensor.dims, dtype=dtype)
    if np.iscomplexobj(ndarray):
        ndarray[tuple(qg8_tensor.indices)] = np.asfortranarray(qg8_tensor.re)\
                                             + 1j*np.asfortranarray(qg8_tensor.im)
    else:
        ndarray[tuple(qg8_tensor.indices)] = np.asfortranarray(qg8_tensor.re)

    return ndarray


# Process tensor nodes
def input(qg8_node):
    qg8_node.outdata = to_numpy(qg8_node.tensor)

def adjacencymatrix(qg8_node): input(qg8_node)

def constant(qg8_node): input(qg8_node)

def ket(qg8_node): input(qg8_node)

def operator(qg8_node): input(qg8_node)

def observable(qg8_node): input(qg8_node)

def time(qg8_node): input(qg8_node)

def track(qg8_node): input(qg8_node)

def output(qg8_node): join(qg8_node)

type_registry = {}

def register_type(node_type, func):
    """bidirectional dictionary"""
    type_registry[node_type] = func
    type_registry[func.__name__] = node_type

register_type(QG8_TYPE_INPUT, input)
register_type(QG8_TYPE_CONSTANT, constant)
register_type(QG8_TYPE_KET, ket)
register_type(QG8_TYPE_OPERATOR, operator)
register_type(QG8_TYPE_OBSERVABLE, observable)
register_type(QG8_TYPE_TIME, time)
register_type(QG8_TYPE_TRACK, track)
register_type(QG8_TYPE_MATMUL, matmul)
register_type(QG8_TYPE_JOIN, join)
register_type(QG8_TYPE_EXPECTATIONVALUE, expectationvalue)
register_type(QG8_TYPE_SOLVEQUTIP, solvequtip)
register_type(QG8_TYPE_MULTIPLY, multiply)
register_type(QG8_TYPE_OUTPUT, output)
