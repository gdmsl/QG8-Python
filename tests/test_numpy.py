#!/usr/bin/python3
# test_numpy.py
# testing script for writing various numpy tensors to QG8 files using
# python -m pytest -rP tests/test_numpy.py
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
from qg8.core import *
from qg8.constants import *
from qg8 import from_numpy

graph = qg8_graph_create()
tests = []

# TEST 1
data = 324
dtype = 'int64'
test = {"description": "TEST1: Constant integer saved as rank 1 tensor " + dtype,
        "first_val": np.array(data, dtype),
        "last_val": np.array(data, dtype),
        "string_id": "constant",
        "dtype": dtype,
        "itype": 'uint8',
        "dims": np.atleast_1d(data).shape,
        "bytes": 16 + 8 + 2
        }

chunk = qg8_chunk_create(QG8_TYPE_CONSTANT, tensor=from_numpy(data, dtype=dtype), string_id=test["string_id"])
qg8_graph_add_chunk(graph, chunk)
tests.append(test)


# TEST 2
data = np.random.randint(0, 2, size=2**16).astype('bool')
dtype = 'uint8'
test = {"description": "TEST2: large 1D boolean array (2**16 elements) saved as sparse tensor " + dtype,
        "first_val": np.array(data[data!=0].item(0), dtype),
        "last_val": np.array(data[data!=0].item(-1), dtype),
        "string_id": "1D sparse array",
        "dtype": dtype,
        "itype": 'uint32',
        "dims": data.shape,
        "bytes": 16 + 1*np.count_nonzero(data) + 4*(np.count_nonzero(data)+1)*len(data.shape)
        }

chunk = qg8_chunk_create(QG8_TYPE_CONSTANT, tensor=from_numpy(data), string_id=test["string_id"])
qg8_graph_add_chunk(graph, chunk)
tests.append(test)


# TEST 3
data = np.zeros((2**8, 2**8))
dtype = 'float32'
test = {"description": "TEST3: large 2D array of zeros (256 x 256 elements) saved as dense tensor " + dtype,
        "first_val": np.array(data.item(0), dtype),
        "last_val": np.array(data.item(-1), dtype),
        "string_id": "2D dense array",
        "dtype": dtype,
        "itype": 'uint16',
        "dims": data.shape,
        "bytes": 16 + 4*data.size + 2*(data.size+1)*len(data.shape)
        }
chunk = qg8_chunk_create(QG8_TYPE_CONSTANT, tensor=from_numpy(data, dtype=dtype, packing=QG8_PACKING_FULL), string_id=test["string_id"])
qg8_graph_add_chunk(graph, chunk)
tests.append(test)


# TEST 4
data = np.random.randint(0, 2**16-1, size=(1, 2, 3, 4, 5, 6))
dtype = 'uint16'
test = {"description": "TEST4: rank 6 tensor saved in sparse format " + dtype + ", without a label",
        "first_val": np.array(data[data!=0].item(0), dtype),
        "last_val": np.array(data[data!=0].item(-1), dtype),
        "string_id": None,
        "dtype": dtype,
        "itype": 'uint8',
        "dims": data.shape,
        "bytes": 16 + 2*np.count_nonzero(data) + 1*(np.count_nonzero(data)+1)*len(data.shape)
        }
chunk = qg8_chunk_create(QG8_TYPE_CONSTANT, tensor=from_numpy(data, dtype=dtype), string_id=test["string_id"])
qg8_graph_add_chunk(graph, chunk)
tests.append(test)


# TEST 5
data = np.random.randint(-1,1,size=(64, 64))+1j*np.random.randint(-1,1,size=(64, 64))
dtype = 'complex128'
test = {"description": "TEST5: complex tensor saved in sparse format " + dtype + ", with dangerous label",
        "first_val": np.array(data[data!=0].item(0), dtype),
        "last_val": np.array(data[data!=0].item(-1), dtype),
        "string_id": "~this\label%is!too long",
        "dtype": dtype,
        "itype": 'uint8',
        "dims": data.shape,
        "bytes": 16 + 16*np.count_nonzero(data) + 1*(np.count_nonzero(data)+1)*len(data.shape)
        }
chunk = qg8_chunk_create(QG8_TYPE_INPUT, tensor=from_numpy(data, dtype=dtype), string_id=test["string_id"])
qg8_graph_add_chunk(graph, chunk)
tests.append(test)


# write all chunks to file and reload it as a new graph
qg8_graph_write('tests/test_numpy.qg8', graph)

graph_new = qg8_graph_load('tests/test_numpy.qg8')


# run module
def run_test(t,chunk):
    print("")
    print(t["description"])

    print("first value: ", end="")
    if chunk.tensor.dtype_id in (QG8_DTYPE_COMPLEX64, QG8_DTYPE_COMPLEX128):
        first_val = chunk.tensor.re[0]+1j*chunk.tensor.im[0]
    else:
        first_val = chunk.tensor.re[0]
    print(t["first_val"], "-> ", first_val)
    assert t["first_val"] == first_val

    print("last value:  ", end="")
    if chunk.tensor.dtype_id in (QG8_DTYPE_COMPLEX64, QG8_DTYPE_COMPLEX128):
        last_val = chunk.tensor.re[-1]+1j*chunk.tensor.im[-1]
    else:
        last_val = chunk.tensor.re[-1]
    print(t["last_val"], "-> ", last_val)
    assert t["last_val"] == last_val

    print("dtype:       ", end="")
    print(t["dtype"], "-> ", chunk.tensor.dtype)
    assert t["dtype"] == chunk.tensor.dtype

    print("itype:       ", end="")
    print(t["itype"], "-> ", chunk.tensor.itype)
    assert t["itype"] == chunk.tensor.itype

    print("string_id:   ", end="")
    print(t["string_id"], "->", chunk.string_id)
    if t["string_id"] is not None:
        assert t["string_id"][0:16] == chunk.string_id
    else:
        assert t["string_id"] == chunk.string_id

    print("dims:        ", end="")
    print(t["dims"], "->", chunk.tensor.dims)
    assert t["dims"] == chunk.tensor.dims

    print("bytes:       ", end="")
    print(t["bytes"], "->", chunk.tensor.datasize())
    assert t["bytes"] == chunk.tensor.datasize()

# run all tests
def test_1():
     run_test(tests[0], graph_new.chunks[0])

def test_2():
    run_test(tests[1], graph_new.chunks[1])

def test_3():
    run_test(tests[2], graph_new.chunks[2])

def test_4():
    run_test(tests[3], graph_new.chunks[3])

def test_5():
    run_test(tests[4], graph_new.chunks[4])

test_1()

test_2()

test_3()

test_4()

test_5()
