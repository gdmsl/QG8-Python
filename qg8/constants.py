#!/usr/bin/python3
# constants.py
# QG8 constants (python)
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

QG8_MAGIC = "QG8XXXXX"
QG8_VERSION = 2

QG8_MODE_READ = 1
QG8_MODE_WRITE = 2

QG8_FLAG_LABEL = 1

QG8_DTYPE_BOOL = 1
QG8_DTYPE_CHAR = 2
QG8_DTYPE_UINT8 = 3
QG8_DTYPE_UINT16 = 4
QG8_DTYPE_UINT32 = 5
QG8_DTYPE_UINT64 = 6
QG8_DTYPE_INT8 = 7
QG8_DTYPE_INT16 = 8
QG8_DTYPE_INT32 = 9
QG8_DTYPE_INT64 = 10
QG8_DTYPE_FLOAT32 = 11
QG8_DTYPE_FLOAT64 = 12
QG8_DTYPE_COMPLEX64 = 13
QG8_DTYPE_COMPLEX128 = 14

QG8_PACKING_FULL_ROW = 1
QG8_PACKING_FULL_COL = 2
QG8_PACKING_SPARSE_COO = 3
QG8_PACKING_HALFHERMITIAN = 4

QG8_TYPE_ADJACENCY = 1
QG8_TYPE_INPUT = 2
QG8_TYPE_CONSTANT = 3
QG8_TYPE_KET = 4
QG8_TYPE_OPERATOR = 5
QG8_TYPE_OBSERVABLE = 6
QG8_TYPE_TIME = 7
QG8_TYPE_TRACK = 8
QG8_TYPE_NOISESPEC = 9
QG8_TYPE_ADD = 10
QG8_TYPE_SUBTRACT = 11
QG8_TYPE_MATMUL = 12
QG8_TYPE_JOIN = 13
QG8_TYPE_SOLVE = 14
QG8_TYPE_EXPECTATIONVALUE = 15
QG8_TYPE_SAMPLE = 16
QG8_TYPE_MULTIPLY = 17
QG8_TYPE_OUTPUT = 19

QG8_TYPE_SOLVEQUTIP = 33

# packing types for extension library
packing_registry = {'full_row_major': QG8_PACKING_FULL_ROW,
                    'full_col_major': QG8_PACKING_FULL_COL,
                    'sparse_coo': QG8_PACKING_SPARSE_COO,
                    'half-hermitian': QG8_PACKING_HALFHERMITIAN}


def dtype_to_name(dtype_id:int):
    if dtype_id == QG8_DTYPE_UINT8:
        return 'uint8'
    if dtype_id == QG8_DTYPE_UINT16:
        return 'uint16'
    if dtype_id == QG8_DTYPE_UINT32:
        return 'uint32'
    if dtype_id == QG8_DTYPE_UINT64:
        return 'uint64'
    if dtype_id == QG8_DTYPE_INT8:
        return 'int8'
    if dtype_id == QG8_DTYPE_INT16:
        return 'int16'
    if dtype_id == QG8_DTYPE_INT32:
        return 'int32'
    if dtype_id == QG8_DTYPE_INT64:
        return 'int64'
    if dtype_id == QG8_DTYPE_FLOAT32:
        return 'float32'
    if dtype_id == QG8_DTYPE_FLOAT64:
        return 'float64'
    if dtype_id == QG8_DTYPE_COMPLEX64:
        return 'complex64'
    if dtype_id == QG8_DTYPE_COMPLEX128:
        return 'complex128'

def dtype_to_char(dtype_id:int):
    if dtype_id == QG8_DTYPE_UINT8:
        return 'B'
    if dtype_id == QG8_DTYPE_UINT16:
        return 'H'
    if dtype_id == QG8_DTYPE_UINT32:
        return 'I'
    if dtype_id == QG8_DTYPE_UINT64:
        return 'Q'
    if dtype_id == QG8_DTYPE_INT8:
        return 'b'
    if dtype_id == QG8_DTYPE_INT16:
        return 'h'
    if dtype_id == QG8_DTYPE_INT32:
        return 'i'
    if dtype_id == QG8_DTYPE_INT64:
        return 'q'
    if dtype_id == QG8_DTYPE_FLOAT32:
        return 'f'
    if dtype_id == QG8_DTYPE_FLOAT64:
        return 'd'
    if dtype_id == QG8_DTYPE_COMPLEX64:
        return 'f'
    if dtype_id == QG8_DTYPE_COMPLEX128:
        return 'd'

