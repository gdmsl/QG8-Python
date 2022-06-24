#!/usr/bin/python3
# core.py
# QG8 core library (python)
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

import struct
from array import array
from qg8.constants import *

#####################
# Data structures
class qg8_file():
    def __init__(self, f, mode):
        self.fp = f
        self.num_chunks = 0
        self.chunks = []
        self.mode = mode


class qg8_chunk():
    """
    # Parent class for qg8 nodes
    """
    def __init__(self, type, flags=0, string_id=None, tensor=None):
        self.type = type
        self.flags = flags
        self.string_id = string_id

        self.tensor = tensor

        if self.string_id is not None:
            self.flags |= QG8_FLAG_LABEL

    def datasize(self):
        """
        Size of the chunk in bytes
        """
        labelled = self.flags & QG8_FLAG_LABEL == QG8_FLAG_LABEL
        if self.tensor is None:
            return 16 + 16 * labelled
        else:
            return 16 + 16 * labelled + self.tensor.datasize()


class qg8_graph():
    """
    Parent class for QG8 graphs as a container for a collection of qg8_node() objects.
    """
    def __init__(self, adj_chunk, chunks=[]):
        self.adj = adj
        self.chunks = chunks

    def datasize(self):
        """
        Size of the graph in bytes
        """
        return self.adj.datasize() + sum([chunk.datasize() for chunk in self.chunks]) + 16


class qg8_iter():
    pass


class qg8_tensor():
    """
    QG8 tensor base class

    args:
      indices : tuple of indices for non-zero elements of the tensor
      re: array, real values
      im: array, imaginary values (None for real tensors)
      num_elements: int, number of stored elements
      dims: tuple, dimensions of the tensor
      rank: int, tensor rank <2**16
      packing: int, packing code
      itype_id: int, typecode for index data
      dtype_id: int, typecode for value data
    """
    def __init__(self, indices, re, im, num_elements, dims, rank, packing: int, itype_id: int, dtype_id: int):
        self.packing = packing
        self.itype_id = itype_id
        self.dtype_id = dtype_id
        self.rank = rank
        self.dims = dims
        self.num_elements = num_elements

        self.indices = indices
        self.re = re
        self.im = im

    @property
    def itype(self):
        return dtype_to_name(self.itype_id)

    @property
    def dtype(self):
        return dtype_to_name(self.dtype_id)

    def __repr__(self):
        return "<QG8 tensor: " \
               + "shape=" + str(self.dims) \
               + ", length=" + str(self.num_elements) \
               + ", itype=" + self.itype \
               + ", dtype=" + self.dtype \
               + ">"

    def datasize(self):
        """
        Size of the tensor in bytes
        """
        return 16 + self.num_elements * _dtype_to_size(self.dtype_id) \
               + (self.num_elements + 1) * self.rank * _dtype_to_size(self.itype_id)


####
# File I/O operations
####
def qg8_file_open(filename: str, mode: int):
    """
    Open a QG8 file for writing or reading

    args:
        filename: str, filename to open
        mode: int, mode in which to open file. Possible values: QG8_MODE_READ or QG8_MODE_WRITE
    """
    if mode == QG8_MODE_READ:
        fp = open(filename, "rb")
    elif mode == QG8_MODE_WRITE:
        fp = open(filename, "wb")
    else:
        raise ValueError("Invalid read/write mode")

    if mode == QG8_MODE_READ and not _integrity_check(fp):
        fp.close()
        raise OSError("Not a valid QG8 file")

    else:
        qg8f = qg8_file(fp, mode)
        return qg8f


def qg8_file_flush(qg8f: qg8_file):
    """
    Write a qg8_file to disk
    """
    if not isinstance(qg8f, qg8_file):
        raise TypeError("Argument is not a qg8_file")
    if qg8f.mode != QG8_MODE_WRITE:
        raise OSError("File is not open in write mode")

    f = qg8f.fp

    # Write header [16 bytes]
    f.write(str.encode(QG8_MAGIC))                                   # Write signature     char[8]
    f.write(_uint(QG8_VERSION, QG8_DTYPE_UINT16))                    # Write version	      uint_16
    f.write(b"\x00\x00\x00\x00\x00\x00")                             # Reserved            6 bytes

    # Iterate over chunks
    for chunk in qg8f.chunks:

        # Write chunk_header [16 bytes + 16 bytes string, incl. skip]
        f.write(_uint(chunk.type, QG8_DTYPE_UINT16))                 # type, uint_16
        labelled = chunk.flags & QG8_FLAG_LABEL == QG8_FLAG_LABEL
        f.write(_uint(chunk.flags, QG8_DTYPE_UINT8))                 # flag byte, uint_8

        if labelled:
            string_id = chunk.string_id.ljust(16, '\x00')[0:16]
            f.write(str.encode(string_id))                          # string_id, char[16]

        f.write(b"\x00\x00\x00\x00\x00")                            # Reserved, 5bytes

        if hasattr(chunk, 'tensor') and chunk.tensor != None:
            tensor = chunk.tensor

            # calculate skip
            d = _dtype_to_size(tensor.dtype_id)  # number of bytes per value
            i = _dtype_to_size(tensor.itype_id)  # number of bytes per index
            iscomplex = (tensor.dtype_id == QG8_DTYPE_COMPLEX64 or tensor.dtype_id == QG8_DTYPE_COMPLEX128)
            num_elements = tensor.num_elements
            rank = tensor.rank

            skip = 16 + num_elements*d + (num_elements+1)*rank*i  # remaining number of bytes in chunk (after skip)
        else:
            skip = 0

        f.write(_uint(skip, QG8_DTYPE_UINT64))                    # bytes to chunk end uint_64

        # write tensor header [16 bytes + dims] and data
        if skip != 0:  # check if there is tensor data

            f.write(_uint(tensor.packing, QG8_DTYPE_UINT8))       # packing type,      uint_8
            f.write(_uint(tensor.itype_id, QG8_DTYPE_UINT8))      # index typecode,    uint_8
            f.write(_uint(tensor.dtype_id, QG8_DTYPE_UINT8))      # data typecode,     uint_8
            f.write(_uint(tensor.rank, QG8_DTYPE_UINT16))         # tensor rank        uint_8

            f.write(b"\x00\x00\x00")                              # Reserved 	       3 bytes

            for m in tensor.dims:
                f.write(_uint(m, tensor.itype_id))               # tensor dimensions m,n,... / uint_d[rank]

            f.write(_uint(num_elements, QG8_DTYPE_UINT64))       # number of elements uint_64

            # Write qg8_tensor_data
            for d in range(tensor.rank):
                if tensor.indices[d].typecode != dtype_to_char(tensor.itype_id):  # confirm itype
                    tensor.indices[d] = array(dtype_to_char(tensor.itype_id), tensor.indices[d])

                f.write(tensor.indices[d].tobytes())

            if tensor.re.typecode != dtype_to_char(tensor.dtype_id):  # confirm dtype
                tensor.re = array(dtype_to_char(tensor.dtype_id), tensor.re)

            f.write(tensor.re.tobytes())

            if iscomplex:
                if tensor.im.typecode != dtype_to_char(tensor.dtype_id):  # confirm dtype
                    tensor.im = array(dtype_to_char(tensor.dtype_id), tensor.im)
                f.write(tensor.im.tobytes())
    return 1


def qg8_file_close(qg8f: qg8_file):
    if not isinstance(qg8f, qg8_file):
        raise TypeError("Argument is not a qg8_file")

    qg8f.fp.close()
    qg8f.mode = 0  # file closed
    return 1


def qg8_file_write_chunk(f: qg8_file, chunk: qg8_chunk):
    """
    Prepares a chunk for writing to a file

    args:
        f: qg8_file, a qg8_file open in write mode
        chunk: qg8_chunk, a qg8_chunk to be written
    """
    if not isinstance(f, qg8_file):
        raise TypeError("First argument is not a qg8_file")
    if not isinstance(chunk, qg8_chunk):
        raise TypeError("Second argument is not a qg8_chunk")
    if f.mode != QG8_MODE_WRITE:
        raise OSError("File is not open in write mode")

    try:
        f.chunks.append(chunk)
        return 1
    except:
        raise OSError("Could not append chunk to qg8_file")


####
# I/O iterator operations
####
def qg8_file_iterator(qg8f: qg8_file):
    """
    Return a new iterator for a given file.
    Used to iteratively load a collection of chunks from a file into memory.

    args:
        f: qg8_file, a qg8_file open in read mode
    """
    if not isinstance(qg8f, qg8_file):
        raise TypeError("Argument is not a qg8_file")
    if qg8f.mode != QG8_MODE_READ:
        raise OSError("File is not open in read mode")

    i = qg8_iter()
    i.offset = 16  # sizeof qg8_file_header
    i.f = qg8f
    return i


def qg8_file_has_next(i: qg8_iter):
    """
    Checks if a file iterator has a next chunk.
    returns 1 if a chunk may be read from an iterator, 0 if EOF.
    """
    if not isinstance(i, qg8_iter):
        raise TypeError("Argument is not a qg8_iter")
    if i.f.mode != QG8_MODE_READ:
        raise IOError("Iterator points to a file which is not open in read mode")

    f = i.f.fp  # get file pointer

    if f.read(1) == b'':  # end of file check
        return 0

    # seek back to start of the chunk without advancing the iterator
    f.seek(i.offset)

    return 1


def qg8_file_next(i: qg8_iter):
    """
    Advances a file iterator to the next chunk.
    returns 1 if a chunk may be read from an iterator, 0 if EOF.
    """
    if not isinstance(i, qg8_iter):
        raise TypeError("Argument is not a qg8_iter")
    if i.f.mode != QG8_MODE_READ:
        raise IOError("Iterator points to a file which is not open in read mode")

    f = i.f.fp  # get file pointer

    if f.read(1) == b'':  # end of file check
        return 0
    else:  # seek to start of the chunk
        f.seek(i.offset)

        _read_int(f, 2)  # chunk type
        flags = _read_int(f, 1)

        labelled = (flags & QG8_FLAG_LABEL) == QG8_FLAG_LABEL
        if labelled:
            f.read(16)

        f.read(5)  # reserved bytes

        skip = _read_int(f, 8)

        i.offset += _sizeof_header(labelled) + skip

    return 1


def qg8_file_extract(iter: qg8_iter):
    """
    Reads a chunk from a file using an iterator
    """
    if not isinstance(iter, qg8_iter):
        raise TypeError("Argument is not a qg8_iter")
    if iter.f.mode != QG8_MODE_READ:
        raise IOError("Iterator points to a file which is not open in read mode")

    # read node header
    f = iter.f.fp  # get file pointer

    # seek to start of the current chunk
    f.seek(iter.offset)

    chunk_type = _read_int(f, 2)

    flags = _read_int(f, 1)

    labelled = (flags & QG8_FLAG_LABEL) == QG8_FLAG_LABEL
    if labelled:
        string_id = f.read(16).decode('ascii').strip('\x00')
    else:
        string_id = None

    reserved = f.read(5)

    skip = _read_int(f, 8)

    # read tensor header
    if skip == 0:
        tensor = None

    else:

        packing = _read_int(f, 1)

        itype_id = _read_int(f, 1)
        dtype_id = _read_int(f, 1)
        iscomplex = (dtype_id == QG8_DTYPE_COMPLEX64 or dtype_id == QG8_DTYPE_COMPLEX128)

        # get typecode and bytes
        itypecode = dtype_to_char(itype_id)
        dtypecode = dtype_to_char(dtype_id)
        i = _dtype_to_size(itype_id)
        d = _dtype_to_size(dtype_id)

        rank = _read_int(f, 2)

        reserved = f.read(3)

        dims = tuple([_read_int(f, i) for j in range(rank)])

        num_elements = _read_int(f, 8)

        indices = []
        for j in range(rank):
            a = array(itypecode, f.read(num_elements * i))
            indices.append(a)

        if iscomplex:
            re = array(dtypecode, f.read(d // 2 * num_elements))
            im = array(dtypecode, f.read(d // 2 * num_elements))
        else:
            re = array(dtypecode, f.read(d * num_elements))
            im = None

        # create a tensor
        tensor = qg8_tensor(indices, re, im, num_elements, dims, rank, packing, itype_id, dtype_id)

    # create a chunk
    chunk = qg8_chunk(chunk_type, flags, string_id, tensor)

    iter.offset += _sizeof_header(labelled) + skip

    return chunk


def _sizeof_header(labelled):
    return 16 + 16 * labelled


####
# Chunk operations
####
def qg8_chunk_create(type, flags=0, string_id=None, tensor=None):
    """
    Creates a new qg8_chunk

    args:
        type: int, type code for the chunk, e.g. QG8_TYPE_INPUT
        flags: int, flag byte for optional settings

        string_id: str, (optional) string label for the chunk
        tensor: qg8_tensor, (optional) qg8 tensor object

    If string_id is not None, the least significant bit of flags will be automatically set to 1
    If string_id is >16 characters long, only the first 16 characters will be written to file
    """
    return qg8_chunk(type, flags, string_id, tensor)


def qg8_chunk_destroy(chunk: qg8_chunk):
    """
    Does nothing. Use `del chunk` to delete a chunk
    """
    if not isinstance(chunk, qg8_chunk):
        raise TypeError("Argument is not a qg8_chunk")
    return 0  # not destroyed


def qg8_chunk_get_tensor(chunk: qg8_chunk):
    """
    Return the tensor that belongs to a chunk. 
    If there is no tensor in the chunk it returns None
    """
    if not isinstance(chunk, qg8_chunk):
        raise TypeError("Argument is not a qg8_chunk")

    if hasattr(chunk, 'tensor'):
        return chunk.tensor
    else:
        return None


def qg8_chunk_get_string_id(chunk: qg8_chunk):
    """
    Return the string_id that belongs to a chunk. 
    If there is no string_id in the chunk it returns None
    """
    if not isinstance(chunk, qg8_chunk):
        raise TypeError("Argument is not a qg8_chunk")
    if hasattr(chunk, 'string_id'):
        return chunk.string_id
    else:
        return None


def qg8_chunk_get_type(chunk: qg8_chunk):
    """
    Return the chunk type as an integer
    """
    if not isinstance(chunk, qg8_chunk):
        raise TypeError("Argument is not a qg8_chunk")
    return chunk.type


def qg8_chunk_get_flags(chunk: qg8_chunk):
    """
    Return the chunk type as an integer
    """
    if not isinstance(chunk, qg8_chunk):
        raise TypeError("Argument is not a qg8_chunk")
    return chunk.flags


####
# Tensor operations
####
def _check_tensor(indices, re, dims, rank):
    if indices is None or re is None or dims is None:
        raise ValueError("Invalid arguments")
    if rank < 1 or rank >= (1 << 16):
        raise ValueError("Invalid tensor rank")
    if any([d < 1 or d >= (1 << 64) for d in dims]):
        raise ValueError("Invalid tensor dimensions")
    return True


def qg8_tensor_create_float(indices: list, re: array, im: array, num_elements: int, dims: list, rank: int, packing: int):
    """
    Create a qg8 tensor from floating point (single precision) values

    args:
        indices: list, list of index arrays
        re: array (float), an array of real values
        im: array (float), (optional) imaginary values.
        num_elements: int, number of tensor entries. Should be equal to len(re)
        dims: list (int), list of tensor dimensions
        rank: int, tensor rank. Should be equal to len(dims)
        packing: int, packing code for the tensor
    """
    _check_tensor(indices, re, dims, rank)

    itype_id = _num_bytes(max(dims))
    if im is not None:
        dtype_id = QG8_DTYPE_COMPLEX64
    else:
        dtype_id = QG8_DTYPE_FLOAT32

    if not isinstance(re, array) or re.typecode != dtype_to_char(dtype_id):
        raise TypeError("re is not an array of type float (single precision)")
    if im is not None:
        if not isinstance(im, array) and im.typecode != dtype_to_char(dtype_id):
            raise TypeError("im is not an array of type float (single precision)")


    return qg8_tensor(indices, re, im, num_elements, dims, rank, packing, itype_id, dtype_id)


def qg8_tensor_create_double(indices: list, re: array, im: array, num_elements: int, dims: list, rank: int, packing: int):
    """
    Create a qg8 tensor from double precision values

    args:
        indices: list, list of index arrays
        re: array (double), an array of real values
        im: array (double), (optional) imaginary values.
        num_elements: int, number of tensor entries. Should be equal to len(re)
        dims: list (int), list of tensor dimensions
        rank: int, tensor rank. Should be equal to len(dims)
        packing: int, packing code for the tensor
    """
    _check_tensor(indices, re, dims, rank)

    itype_id = _num_bytes(max(dims))
    if im is not None:
        dtype_id = QG8_DTYPE_COMPLEX128
    else:
        dtype_id = QG8_DTYPE_FLOAT64

    if not isinstance(re, array) or re.typecode != dtype_to_char(dtype_id):
        raise TypeError("re is not an array of type double")
    if im is not None:
        if not isinstance(im, array) and im.typecode != dtype_to_char(dtype_id):
            raise TypeError("im is not an array of type double")

    return qg8_tensor(indices, re, im, num_elements, dims, rank, packing, itype_id, dtype_id)


def qg8_tensor_create_uint8(indices: list, re: array, num_elements: int, dims: list, rank: int, packing: int):
    """
    Create a qg8 tensor from 8 bit unsigned integer values

    args:
        indices: list, list of index arrays
        re: array (uint8), an array of real values
        num_elements: int, number of tensor entries. Should be equal to len(re)
        dims: list (int), list of tensor dimensions
        rank: int, tensor rank. Should be equal to len(dims)
        packing: int, packing code for the tensor
    """
    _check_tensor(indices, re, dims, rank)

    itype_id = _num_bytes(max(dims))
    dtype_id = QG8_DTYPE_UINT8

    if not isinstance(re, array) or re.typecode != dtype_to_char(dtype_id):
        raise TypeError("re is not an array of type uint8")

    return qg8_tensor(indices, re, None, num_elements, dims, rank, packing, itype_id, dtype_id)


def qg8_tensor_create_uint16(indices: list, re: array, num_elements: int, dims: list, rank: int, packing: int):
    """
    Create a qg8 tensor from 16 bit unsigned integer values

    args:
        indices: list, list of index arrays
        re: array (uint16), an array of real values
        num_elements: int, number of tensor entries. Should be equal to len(re)
        dims: list (int), list of tensor dimensions
        rank: int, tensor rank. Should be equal to len(dims)
        packing: int, packing code for the tensor
    """
    _check_tensor(indices, re, dims, rank)

    itype_id = _num_bytes(max(dims))
    dtype_id = QG8_DTYPE_UINT16

    if not isinstance(re, array) or re.typecode != dtype_to_char(dtype_id):
        raise TypeError("re is not an array of type uint16")

    return qg8_tensor(indices, re, None, num_elements, dims, rank, packing, itype_id, dtype_id)


def qg8_tensor_create_uint32(indices: list, re: array, num_elements: int, dims: list, rank: int, packing: int):
    """
    Create a qg8 tensor from 32 bit unsigned integer values

    args:
        indices: list, list of index arrays
        re: array (uint32), an array of real values
        num_elements: int, number of tensor entries. Should be equal to len(re)
        dims: list (int), list of tensor dimensions
        rank: int, tensor rank. Should be equal to len(dims)
        packing: int, packing code for the tensor
    """
    _check_tensor(indices, re, dims, rank)

    itype_id = _num_bytes(max(dims))
    dtype_id = QG8_DTYPE_UINT32

    if not isinstance(re, array) or re.typecode != dtype_to_char(dtype_id):
        raise TypeError("re is not an array of type uint32")

    return qg8_tensor(indices, re, None, num_elements, dims, rank, packing, itype_id, dtype_id)


def qg8_tensor_create_uint64(indices: list, re: array, num_elements: int, dims: list, rank: int, packing: int):
    """
    Create a qg8 tensor from 64 bit unsigned integer values

    args:
        indices: list, list of index arrays
        re: array (uint64), an array of real values
        num_elements: int, number of tensor entries. Should be equal to len(re)
        dims: list (int), list of tensor dimensions
        rank: int, tensor rank. Should be equal to len(dims)
        packing: int, packing code for the tensor
    """
    _check_tensor(indices, re, dims, rank)

    itype_id = _num_bytes(max(dims))
    dtype_id = QG8_DTYPE_UINT64

    if not isinstance(re, array) or re.typecode != dtype_to_char(dtype_id):
        raise TypeError("re is not an array of type uint64")

    return qg8_tensor(indices, re, None, num_elements, dims, rank, packing, itype_id, dtype_id)


def qg8_tensor_create_int8(indices: list, re: array, num_elements: int, dims: list, rank: int, packing: int):
    """
    Create a qg8 tensor from 8 bit integer values

    args:
        indices: list, list of index arrays
        re: array (int8), an array of real values
        num_elements: int, number of tensor entries. Should be equal to len(re)
        dims: list (int), list of tensor dimensions
        rank: int, tensor rank. Should be equal to len(dims)
        packing: int, packing code for the tensor
    """
    _check_tensor(indices, re, dims, rank)

    itype_id = _num_bytes(max(dims))
    dtype_id = QG8_DTYPE_INT8

    if not isinstance(re, array) or re.typecode != dtype_to_char(dtype_id):
        raise TypeError("re is not an array of type int8")

    return qg8_tensor(indices, re, None, num_elements, dims, rank, packing, itype_id, dtype_id)


def qg8_tensor_create_int16(indices: list, re: array, num_elements: int, dims: list, rank: int, packing: int):
    """
    Create a qg8 tensor from 16 bit integer values

    args:
        indices: list, list of index arrays
        re: array (int16), an array of real values
        num_elements: int, number of tensor entries. Should be equal to len(re)
        dims: list (int), list of tensor dimensions
        rank: int, tensor rank. Should be equal to len(dims)
        packing: int, packing code for the tensor
    """
    _check_tensor(indices, re, dims, rank)

    itype_id = _num_bytes(max(dims))
    dtype_id = QG8_DTYPE_INT16

    if not isinstance(re, array) or re.typecode != dtype_to_char(dtype_id):
        raise TypeError("re is not an array of type int16")

    return qg8_tensor(indices, re, None, num_elements, dims, rank, packing, itype_id, dtype_id)


def qg8_tensor_create_int32(indices: list, re: array, num_elements: int, dims: list, rank: int, packing: int):
    """
    Create a qg8 tensor from 32 bit integer values

    args:
        indices: list, list of index arrays
        re: array (int32), an array of real values
        num_elements: int, number of tensor entries. Should be equal to len(re)
        dims: list (int), list of tensor dimensions
        rank: int, tensor rank. Should be equal to len(dims)
        packing: int, packing code for the tensor
    """
    _check_tensor(indices, re, dims, rank)

    itype_id = _num_bytes(max(dims))
    dtype_id = QG8_DTYPE_INT32

    if not isinstance(re, array) or re.typecode != dtype_to_char(dtype_id):
        raise TypeError("re is not an array of type int32")

    return qg8_tensor(indices, re, None, num_elements, dims, rank, packing, itype_id, dtype_id)


def qg8_tensor_create_int64(indices: list, re: array, num_elements: int, dims: list, rank: int, packing: int):
    """
    Create a qg8 tensor from 64 bit integer values

    args:
        indices: list, list of index arrays
        re: array (int64), an array of real values
        num_elements: int, number of tensor entries. Should be equal to len(re)
        dims: list (int), list of tensor dimensions
        rank: int, tensor rank. Should be equal to len(dims)
        packing: int, packing code for the tensor
    """
    _check_tensor(indices, re, dims, rank)

    itype_id = _num_bytes(max(dims))
    dtype_id = QG8_DTYPE_INT64

    if not isinstance(re, array) or re.typecode != dtype_to_char(dtype_id):
        raise TypeError("re is not an array of type int64")

    return qg8_tensor(indices, re, None, num_elements, dims, rank, packing, itype_id, dtype_id)

def qg8_tensor_destroy(t: qg8_tensor):
    """
    Does nothing. Use `del tensor` to delete a tensor
    """
    if not isinstance(t, qg8_tensor):
        raise ValueError("Argument is not a qg8_tensor")
    return 0  # not destroyed


def qg8_tensor_get_rank(t: qg8_tensor):
    if not isinstance(t, qg8_tensor):
        raise ValueError("Argument is not a qg8_tensor")
    return t.rank


def qg8_tensor_get_dims(t: qg8_tensor):
    if not isinstance(t, qg8_tensor):
        raise ValueError("Argument is not a qg8_tensor")
    return t.dims


def qg8_tensor_get_num_elems(t: qg8_tensor):
    if not isinstance(t, qg8_tensor):
        raise ValueError("Argument is not a qg8_tensor")
    return t.num_elements


def qg8_tensor_get_indices(t: qg8_tensor):
    if not isinstance(t, qg8_tensor):
        raise ValueError("Argument is not a qg8_tensor")
    return t.indices


def qg8_tensor_get_dtypeid(t: qg8_tensor):
    if not isinstance(t, qg8_tensor):
        raise ValueError("Argument is not a qg8_tensor")
    return t.dtype_id


def qg8_tensor_get_itypeid(t: qg8_tensor):
    if not isinstance(t, qg8_tensor):
        raise ValueError("Argument is not a qg8_tensor")
    return t.itype_id


def qg8_tensor_get_re(t: qg8_tensor):
    if not isinstance(t, qg8_tensor):
        raise ValueError("Argument is not a qg8_tensor")
    return t.re


def qg8_tensor_get_im(t: qg8_tensor):
    if not isinstance(t, qg8_tensor):
        raise ValueError("Argument is not a qg8_tensor")
    if not hasattr(t, "im"):
        return None
    return t.im


####
# Graph operations
####
def qg8_graph_load(filename: str):
    """"
    Load a QG8 graph from file chunk by chunk
    """
    f = qg8_file_open(filename, QG8_MODE_READ)
    if f is None:
        return None

    adj = None

    i = qg8_file_iterator(f)
    while qg8_file_has_next(i):
        chunk = qg8_file_extract(i)
        if chunk is None:
            return None
        if qg8_chunk_get_type(chunk) != QG8_TYPE_ADJACENCY:
            continue
        adj = chunk

    if adj == None:
        raise IOError("File has no chunk of ADJACENCY type for graph")

    chunks = []

    while qg8_file_has_next(i):  # EOF check
        chunk = qg8_file_extract(i)
        if chunk is None:
            return None
        chunks.append(chunk)

    qg8_file_close(f)

    return qg8_graph(adj, chunks)


def qg8_graph_write(filename: str, graph: qg8_graph):
    """
    Wrapper function which prepares a collection of chunks (graph) and writes it to a file
    """
    if not isinstance(graph, qg8_graph):
        raise TypeError("Second argument is not a qg8_graph")

    try:
        qg8f = qg8_file_open(filename, QG8_MODE_WRITE)
    except:
        raise IOError("Could not open file in write mode")

    success = 1
    success *= qg8_file_write_chunk(qg8f, graph.adj)

    for chunk in graph.chunks:
        success *= qg8_file_write_chunk(qg8f, chunk)

    qg8_file_flush(qg8f)
    qg8_file_close(qg8f)
    return success


def qg8_graph_create():
    """
    Create an empty qg8_graph
    """
    return qg8_graph(None, [])


def qg8_graph_destroy(graph: qg8_graph):
    """
    Does nothing. Use `del graph` to delete a graph
    """
    return 0  # not destroyed


def qg8_graph_get_number_chunks(graph: qg8_graph):
    """
    Get the total number of chunks contained in the graph
    """
    if not isinstance(graph, qg8_graph):
        raise TypeError("Argument is not a qg8_graph")
    return len(graph.chunks)


def qg8_graph_get_chunk(graph: qg8_graph, idx: int):
    """
    Get a chunk from the graph according to its index provided the second argument `idx`
    """
    if not isinstance(graph, qg8_graph):
        raise TypeError("Argument is not a qg8_graph")
    if idx > len(graph.chunks) - 1:
        raise ValueError("Index exceeds the number of chunks in the graph")
    return graph.chunks[idx]


def qg8_graph_add_chunk(graph: qg8_graph, chunk: qg8_chunk):
    """
    Add a new chunk to the graph
    """
    if not isinstance(graph, qg8_graph):
        raise TypeError("Argument is not a qg8_graph")
    if not isinstance(chunk, qg8_chunk):
        raise TypeError("Argument is not a qg8_chunk")
    graph.chunks.append(chunk)
    return 1


def qg8_graph_remove_chunk(graph: qg8_graph, chunk: qg8_chunk):
    """
    Remove all instances of `chunk` from the graph
    """
    if not isinstance(graph, qg8_graph):
        raise TypeError("Argument is not a qg8_graph")
    if not isinstance(chunk, qg8_chunk):
        raise TypeError("Argument is not a qg8_chunk")
    if len(graph.chunks) == 0:
        raise ValueError("Graph contains no chunks")
    if chunk in graph.chunks:
        graph.chunks = [c for c in graph.chunks if c is not chunk]
        return 1
    else:
        return 0


def _num_bytes(val:int):
    if val < (1 << 8):
        return QG8_DTYPE_UINT8
    elif val < (1 << 16):
        return QG8_DTYPE_UINT16
    elif val < (1 << 32):
        return QG8_DTYPE_UINT32
    else:
        return QG8_DTYPE_UINT64


def _dtype_to_size(dtype_id:int):
    if dtype_id == QG8_DTYPE_UINT8:
        return 1
    if dtype_id == QG8_DTYPE_UINT16:
        return 2
    if dtype_id == QG8_DTYPE_UINT32:
        return 4
    if dtype_id == QG8_DTYPE_UINT64:
        return 8
    if dtype_id == QG8_DTYPE_INT8:
        return 1
    if dtype_id == QG8_DTYPE_INT16:
        return 2
    if dtype_id == QG8_DTYPE_INT32:
        return 4
    if dtype_id == QG8_DTYPE_INT64:
        return 8
    if dtype_id == QG8_DTYPE_FLOAT32:
        return 4
    if dtype_id == QG8_DTYPE_FLOAT64:
        return 8
    if dtype_id == QG8_DTYPE_COMPLEX64:
        return 8
    if dtype_id == QG8_DTYPE_COMPLEX128:
        return 16


def _integrity_check(f):
    """
    Returns True if argument is a valid QG8 file
    """
    sig = f.read(8).decode('ascii')
    version = _read_int(f, 2)
    res = f.read(6)

    if sig[0:3] != QG8_MAGIC[0:3]:
        print("Invalid magic string {}\n".format(sig))
        return False

    elif version < 1 or version > QG8_VERSION:
        print("Unsupported file version {}\n".format(version))
        return False

    return True


def _uint(x, dtype_id):
    """
    Convert integer to binary representation.
    """
    char = dtype_to_char(dtype_id)
    return struct.pack(char, x)


def _read_int(f, b):
    """
    Read an integer of a specified number of bytes from the filestream f
    """
    if b == 1:
        return struct.unpack('<B', f.read(b))[0]
    elif b == 2:
        return struct.unpack('<H', f.read(b))[0]
    elif b == 4:
        return struct.unpack('<I', f.read(b))[0]
    elif b == 8:
        return struct.unpack('<Q', f.read(b))[0]
