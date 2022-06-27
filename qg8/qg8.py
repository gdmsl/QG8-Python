#!/usr/bin/python3
# qg8.py
# QG8 python extension library.
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

import numpy as np
from array import array
import numbers
from qg8.constants import *
import qg8.ops
import qg8.core
from qg8.core import qg8_graph_write as save


class QG8Node(qg8.core.qg8_chunk):
    """
    Base node class for constructing QG8 graphs
    args:
        type: int
    kwargs (optional):
        flags: int
        string_id: str
        tensor: qg8_tensor (only for tensor nodes)

    Can also be initialized directly from a qg8_chunk
    This class defines the attribute input_nodes which is used to compute the graph
    """
    def __init__(self, *args, **kwargs):
        if isinstance(args[0],qg8.core.qg8_chunk):
            # initialize from qg8_chunk object
            self.__dict__ = args[0].__dict__
        else:
            super().__init__(*args, **kwargs)
        self.input_nodes = []


class TensorNode(QG8Node):
    def __array__(self):
        return self.full()

    @property
    def shape(self):
        return self.tensor.dims

    def full(self):
        return qg8.ops.to_numpy(self.tensor)

    def indices(self):
        return [np.array(x) for x in self.tensor.indices]

    def values(self):
        if self.tensor.im is None:
            return np.array(self.tensor.re)
        else:
            return np.array(self.tensor.re) + 1j*np.array(self.tensor.im)

    def __repr__(self):
        return "<QG8 tensor node:" + qg8.ops.type_registry[self.type].__name__ \
               + "{}".format(", string_id='" + self.string_id + "'" if self.string_id is not None else '') \
               + ", shape=" + str(self.tensor.dims) \
               + ", length=" + str(self.tensor.num_elements) \
               + ", itype=" + self.tensor.itype \
               + ", dtype=" + self.tensor.dtype \
               + ">"


class OpNode(QG8Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_nodes = []

    def doc(self):
        print(qg8.ops.type_registry[self.type].__doc__)

    def __repr__(self):
        return "<QG8 op node:" \
               + qg8.ops.type_registry[self.type].__name__ \
               + "{}".format(", string_id='" + self.string_id + "'" if self.string_id is not None else '') \
               + ">"


class AdjacencyMatrix(TensorNode):
    def __repr__(self):
        return "<QG8 adjacency matrix:" \
                + "{}".format(", string_id='" + self.string_id + "'" if self.string_id is not None else '') \
                + ", shape=" + str(self.tensor.dims) \
                + ", length=" + str(self.tensor.num_elements) \
                + ">"


class QuantumGraph(qg8.core.qg8_graph):
    """
    QuantumGraph is a container for a collection of QG8Node() objects.
    It extends on the qg8_graph base class and provides convenience functions for building
    quantum graphs from numpy arrays
    """
    def __init__(self, *args):
        self.chunks = []

        if len(args) > 0:
            self.adj = None
            """Initialize graph from a set of qg8_chunks or nodes"""
            for n in args:
                if isinstance(n, qg8.core.qg8_chunk):
                    if n.tensor is None:
                        self.chunks.append(OpNode(n))
                    elif n.type == QG8_TYPE_ADJACENCY:
                        self.adj = AdjacencyMatrix(n)
                    else:
                        self.chunks.append(TensorNode(n))

            # call build edges
            self.build_edges()

    @property
    def nodes(self):
        """
        Returns a list of nodes in the graph (all chunks except adjacency chunks)
        """
        nodes = [n for n in self.chunks if n.type is not QG8_TYPE_ADJACENCY]
        return nodes

    def __getattr__(self, attr):
        """
        Grab instance method calls and add them to the graph as nodes
        """
        def __handler(*args, **kwargs):
            try:
                type = qg8.ops.type_registry[__handler.__name__]
                return self.add(type, *args, **kwargs)
            except KeyError:
                print("QG8 warning: method '{}' not defined in type_registry. Node not added to the graph".format(
                    __handler.__name__))

        """python magic for catching bound method calls and mapping them to qg8_ops"""
        if attr in qg8.ops.type_registry:
            handler = __handler
            handler.__name__ = attr  # Change the method's name
            return handler
        else: print("QG8 warning: unrecognised method '{}'".format(attr))

    def remove(self, node):
        self.chunks.remove(node)

    def add(self, node_type, *inputs, string_id: str = None, packing=QG8_PACKING_SPARSE_COO, dtype=None):
        """
        master method to add a new node to the graph.
        args:
            node_type: int, unique ID indicating the type of node
            inputs: QG8Node or tensor like objects
        kwargs:
            string_id: str, <16char string
            packing: int, packing code used to unpack tensor data (for tensor-like objects)
            dtype: numpy style string specifying the desired data type, e.g. float32, complex128

        accepted input types:
            QG8Node objects (for op nodes)
            numpy array or constant (tensor node)
            custom tensor-like object (tensor node)
        """
        def _packing_id(packing):
            if isinstance(packing, str):  # convert string to packing code
                if packing in packing_registry:
                    return packing_registry[packing]
            elif isinstance(packing, int):
                return packing
            else:
                raise ValueError("Unrecognised packing type")

        if _is_op_node(inputs):  # op node
            chunk = OpNode(node_type, string_id=string_id)
            chunk.input_nodes = inputs

        else:  # tensor node
            if _is_qg8_tensor(inputs):
                tensor = inputs[0]

            elif _is_numpy_tensor(inputs):  # numpy array
                tensor = from_numpy(inputs[0],
                                    packing=_packing_id(packing), dtype=dtype)

            elif _is_custom_tensor(inputs):  # custom tensor
                tensor = ijv_to_qg8_tensor(inputs[0].indices(), inputs[0].values(), inputs[0].shape,
                                           packing=_packing_id(packing), dtype=dtype)

            else:
                raise TypeError("Unrecognised node or tensor type")

            # create chunk
            if node_type == QG8_TYPE_ADJACENCY:
                chunk = AdjacencyMatrix(node_type, tensor=tensor, string_id=string_id)
            else:
                chunk = TensorNode(node_type, tensor=tensor, string_id=string_id)

        self.chunks.append(chunk)
        return chunk

    def adjacency_matrix(self):
        """
        Build an adjacency matrix from graph nodes in qg8_tensor format
        and add it to the graph as a new chunk
        """
        dim = len(self.chunks)
        itype_id = qg8.core._num_bytes(dim)
        dtype_id = QG8_DTYPE_UINT8

        values = array(dtype_to_char(dtype_id), [])
        indices0 = array(dtype_to_char(itype_id), [])
        indices1 = array(dtype_to_char(itype_id), [])

        for k, n in enumerate(self.chunks):
            if hasattr(n, 'input_nodes') and len(n.input_nodes) > 0:
                values.extend(1 for b in n.input_nodes)
                indices0.extend(self.chunks.index(b) for b in n.input_nodes)
                indices1.extend(k for b in n.input_nodes)

        A = qg8.core.qg8_tensor([indices0, indices1], values, None, len(values), (dim, dim), 2,
                       QG8_PACKING_SPARSE_COO, itype_id, dtype_id)

        self.adj = AdjacencyMatrix(QG8_TYPE_ADJACENCY, flags=0, string_id=None, tensor=A)
        return self.adj

    def build_edges(self):
        """
        Build edges between graph nodes according to the adjacency matrix chunk_adj
        """
        if self.adj is None:
            return

        for n in self.chunks:
            n.input_nodes = []

        for i, k in zip(*chunk_adj.tensor.indices):
            self.chunks[k].input_nodes.append(self.chunks[i])


class Track():
    """
    Custom class that is initialized with a list of time points and returns a list of samples.
    To be used as a tensor node this class should implement the methods indices() and values()
    and have an attribute shape.
    """

    def __init__(self, times):
        self.t = times
        self._samples = np.zeros(len(times))

    def add_sinepulse(self, start=0, duration=0, amplitude=0, overwrite=True):
        """
        generates a sine_pulse and adds it to the track
        params: three element list for the start, duration and amplitude of the pulse
        default behavior is to overwrite preexisting data
        unless overwrite=False then the new pulse will be added to the existing data
        """
        dt = self.t[1] - self.t[0]
        for idx, ti in enumerate(self.t):
            if ti >= start and ti < start + duration:
                value = amplitude * np.sin((ti - start + dt) * np.pi / duration) ** 2
                if overwrite:
                    self._samples[idx] = value
                else:
                    self._samples[idx] += value

    def add_squarepulse(self, start=0, duration=0, amplitude=0, overwrite=True):
        """
        same as add_sinepulse but for a square pulse
        """
        dt = self.t[1] - self.t[0]
        for idx, ti in enumerate(self.t):
            if ti >= start and ti < start + duration:
                value = amplitude
                if overwrite:
                    self._samples[idx] = value
                else:
                    self._samples[idx] += value

    def indices(self):
        return self._samples.nonzero()

    def values(self):
        return self._samples[self._samples != 0]

    @property
    def shape(self):
        return (len(self.t),)


def load(filename: str):
    """
    Read in and initialize a quantum graph.
      uses the qg8_graph_load method in qg8_core
      the returned graph is then converted into a QuantumGraph object
    """
    qg8_graph = qg8.core.qg8_graph_load(filename)

    return QuantumGraph(*qg8_graph.adj, *qg8_graph.chunks)


def np2array(ndarray, dtype_id):
    """
    Helper function to do fast format conversion between numpy array and array.array
    used for internal storage of QG8 tensors
    args:
        ndarray, numpy array of values
        dtype_id, target data type as a QG8 constant
    """
    typecode = dtype_to_char(dtype_id)
    dtype = dtype_to_name(dtype_id)

    return array(typecode, ndarray.astype(dtype).tobytes())


def ijv_to_qg8_tensor(indices, values, dims, packing: int = QG8_PACKING_SPARSE_COO, dtype=None):
    """
    Convert tensor data in ijv format to a qg8_tensor. Uses numpy arrays as an intermediate format for type conversion
    if dtype is not set, it will be automatically determined from the numpy dtype

    args:
        indices: list or tuple of 1D integer arrays for each dimension or iterable array with shape (rank, length)
        values: 1D array of (complex) values (if complex)
        dims: tensor dimensions, shape = (rank,)
    kwargs:
        packing: packing code for the tensor, e.g. QG8_PACKING_FULL
        dtype: numpy.dtype object or string specifying the target data type to store the array.

    Supported dtypes:
        float32, float64, complex64, complex128
        other types will be automatically converted
    """
    _values = np.array(values)  # conversion to numpy array
    _dims = (*dims,)

    if dtype is None:
        dtype = _values.dtype
    else:
        dtype = np.dtype(dtype)

    itype = qg8.core._num_bytes(max(_dims))

    if indices is None:
        if len(_values) != np.product(dims):
            raise ValueError("Number of elements does not matches dimensionality")
        qg8_indices = None
    else:
        _indices = [np.array(x, int) for x in indices]

        if len(_indices) != len(dims) or len(_indices[0]) != len(_values):
            raise ValueError("Bad tensor dimensions or unknown tensor format")

        qg8_indices = [np2array(x, itype) for x in _indices]


    # unsigned integers
    if dtype in ('bool', 'uint8'):
        qg8_re = np2array(_values.flatten().real, QG8_DTYPE_UINT8)  # fast format/type conversion
        return qg8.core.qg8_tensor_create_uint8(qg8_indices, qg8_re, len(qg8_re), _dims, len(_dims), packing)
    elif dtype in ('uint16',):
        qg8_re = np2array(_values.flatten().real, QG8_DTYPE_UINT16)  # fast format/type conversion
        return qg8.core.qg8_tensor_create_uint16(qg8_indices, qg8_re, len(qg8_re), _dims, len(_dims), packing)
    elif dtype in ('uint32',):
        qg8_re = np2array(_values.flatten().real, QG8_DTYPE_UINT32)  # fast format/type conversion
        return qg8.core.qg8_tensor_create_uint32(qg8_indices, qg8_re, len(qg8_re), _dims, len(_dims), packing)
    elif dtype in ('uint64',):
        qg8_re = np2array(_values.flatten().real, QG8_DTYPE_UINT64)  # fast format/type conversion
        return qg8.core.qg8_tensor_create_uint64(qg8_indices, qg8_re, len(qg8_re), _dims, len(_dims), packing)
    # signed integers
    elif dtype in ('int8',):
        qg8_re = np2array(_values.flatten().real, QG8_DTYPE_INT8)  # fast format/type conversion
        return qg8.core.qg8_tensor_create_int8(qg8_indices, qg8_re, len(qg8_re), _dims, len(_dims), packing)
    elif dtype in ('int16',):
        qg8_re = np2array(_values.flatten().real, QG8_DTYPE_INT16)  # fast format/type conversion
        return qg8.core.qg8_tensor_create_int16(qg8_indices, qg8_re, len(qg8_re), _dims, len(_dims), packing)
    elif dtype in ('int32',):
        qg8_re = np2array(_values.flatten().real, QG8_DTYPE_INT32)  # fast format/type conversion
        return qg8.core.qg8_tensor_create_int32(qg8_indices, qg8_re, len(qg8_re), _dims, len(_dims), packing)
    elif dtype in ('int64',):
        qg8_re = np2array(_values.flatten().real, QG8_DTYPE_INT64)  # fast format/type conversion
        return qg8.core.qg8_tensor_create_int64(qg8_indices, qg8_re, len(qg8_re), _dims, len(_dims), packing)

    # float types
    elif dtype in ('float32', 'complex64'):
        qg8_re = np2array(_values.flatten().real, QG8_DTYPE_FLOAT32)  # fast format/type conversion
        if dtype == 'complex64':
            qg8_im = np2array(_values.flatten().imag, QG8_DTYPE_FLOAT32)
        else:
            qg8_im = None

        return qg8.core.qg8_tensor_create_float(qg8_indices, qg8_re, qg8_im, len(qg8_re), _dims, len(_dims), packing)

    # double types
    elif dtype in ('float64', 'complex128'):
        qg8_re = np2array(_values.flatten().real, QG8_DTYPE_FLOAT64)  # fast format/type conversion
        if dtype == 'complex128':
            qg8_im = np2array(_values.flatten().imag, QG8_DTYPE_FLOAT64)
        else:
            qg8_im = None

        return qg8.core.qg8_tensor_create_double(qg8_indices, qg8_re, qg8_im, len(qg8_re), _dims, len(_dims), packing)

    else:
        raise ValueError('Unrecognised dtype' + dtype)


def topology_sort(end_node):
    ordering = []
    visited_nodes = set()

    def recursive_helper(node):
        if len(node.input_nodes) > 0:  # only for op nodes
            for input_node in node.input_nodes:
                if input_node not in visited_nodes:
                    recursive_helper(input_node)

        visited_nodes.add(node)
        ordering.append(node)

    # start recursive search
    recursive_helper(end_node)
    return ordering


def forward(qg8_node):
    qg8.ops.type_registry[qg8_node.type](qg8_node)


def run(qg8_graph, end_node=None):
    if end_node is None:
        end_node = qg8_graph.nodes[-1]
    nodes_sorted = topology_sort(end_node)
    for node in nodes_sorted:
        forward(node)
    return end_node.outdata


def node_types():
    return {k for k in qg8.ops.type_registry if isinstance(k, str)}


def _is_op_node(obj):
    return isinstance(obj[0], qg8.core.qg8_chunk)


def _is_qg8_tensor(obj):
    return isinstance(obj[0], qg8.core.qg8_tensor)


def _is_numpy_tensor(obj):
    return (len(obj) == 1 and
            (isinstance(obj[0], np.ndarray) or
            isinstance(obj[0], numbers.Number))
            )


def from_numpy(ndarray, **kwargs):
    """
    Convert numpy array to qg8_tensor
    """
    ndarray = np.atleast_1d(ndarray)
    dims = list(ndarray.shape)

    if 'packing' not in kwargs or (not _isfullpacking(kwargs['packing'])):
        return ijv_to_qg8_tensor(ndarray.nonzero(), ndarray[ndarray != 0], dims, **kwargs)
    elif kwargs['packing'] == QG8_PACKING_FULL_COL:
        return ijv_to_qg8_tensor(None, ndarray.flatten(order='C'), dims, **kwargs)
    else: # kwargs['packing'] == QG8_PACKING_FULL_ROW
        return ijv_to_qg8_tensor(None, ndarray.flatten(order='F'), dims, **kwargs)


def _is_custom_tensor(obj):
    return (len(obj) == 1 and
            hasattr(obj[0], 'indices') and
            hasattr(obj[0], 'values') and
            hasattr(obj[0], 'shape')
            )
