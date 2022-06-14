# compatibility to future features
from . import future

import numpy as np

from .dataset_writer import DatasetBufferedWriter, DatasetDirectWriter
from .source import Source


class MultiTrainData:
    def __init__(self, count, data):
        self.count = count
        self.data = data


class DataSetterBase:
    def __set__(self, instance, value):
        raise NotImplementedError


class BlockedSetter(DataSetterBase):
    def __set__(self, instance, value):
        raise RuntimeError(
            "Class attributes interface is disabled. Use option "
            "'class_attrs_interface=True' to enable it.")


class MultiTrainDataSetter(DataSetterBase):
    """Overrides the setters for attributes which declared as datasets
    in order to use the assignment operation for adding data in a train
    """
    def __init__(self, name):
        self.name = name

    def __set__(self, instance, value):
        if isinstance(value, MultiTrainData):
            instance.add_value(value.count, self.name, value.data)
        else:
            raise TypeError(f"The attribute '{self.name}' accepts only "
                            "'MultiTrainData' instance")

class SingleTrainDataSetter(DataSetterBase):
    def __init__(self, name):
        self.name = name

    def __set__(self, instance, value):
        instance.add_train_value(self.name, value)


class NextTrainProxy:
    def __init__(self, writer):
        self.writer = writer

    def add_train_value(self, name, value):
        self.writer.add_train_value(name, value)


# Attention! `Dataset` is the descriptor class and its instances are
# intended to be used as class members of `FileWriter` children. Changing
# them leads to changes in the host class itself and in all its instances.
# Therefore, one can change `self` only in the `__init__` method and
# in methods that are called from the `FileWriterMeta` metaclass.
class DatasetBase:
    """Base dataset descriptor class"""
    source_class = None

    class Attributes:
        """Dataset attributes structure"""
        def __init__(self, **kwargs):
            for kw, val in kwargs.items():
                setattr(self, kw, val)

    def __init__(self, source_name, key, entry_shape, dtype,
                 chunks=None, compression=None):
        self.entry_shape = entry_shape
        self.dtype = dtype
        self.compression = compression
        self.chunks = chunks
        if source_name and source_name[0] == '@':
            self.orig_name = (True, source_name[1:], key)
        else:
            self.orig_name = (False, source_name, key)

    def set_name(self, source_name, key):
        """Sets new name to the dataset"""
        self.orig_name = (False, source_name, key)
        self.canonical_name = (source_name, key)

    def resolve_name(self, aliases={}):
        """Normalizes source name and key"""
        isalias, source_id, key = self.orig_name

        # resolve reference
        source_name = aliases[source_id] if isalias else source_id
        self.canonical_name = (source_name, key)

    def get_dataset_fullname(self, writer):
        # expected to return (source_name, key, stype)
        raise NotImplementedError

    def get_entry_attr(self, writer):
        # expected to return (entry_shape, dtype)
        return self.entry_shape, self.dtype

    def get_chunks(self, writer):
        # expected to return chunks
        return self.chunks

    def init_dataset_attr(self, writer):
        # expand names
        source_name, key, stype = self.get_dataset_fullname(writer)

        # expand entry attributes
        entry_shape, dtype = self.get_entry_attr(writer)

        # auto chunking
        chunks = self.get_chunks(writer)
        if chunks is None:
            chunks = Dataset._chunks_autosize(
                writer._meta.max_train_per_file, entry_shape,
                dtype, stype)

        writer._ds_attrs[id(self)] = Dataset.Attributes(
            source_name=source_name, key=key, stype=stype,
            entry_shape=entry_shape, dtype=dtype, chunks=chunks,
            compression=self.compression,
        )

    def check_value(self, writer, value):
        """Checks data"""
        # can we need to checked type cast?

        # shape check
        entry_shape = self(writer).entry_shape
        value_shape = np.shape(value)
        shape = np.broadcast_shapes(value_shape, (1,) + entry_shape)
        if shape == entry_shape:
            nrec = 1
        elif shape[1:] == entry_shape:
            nrec = shape[0]
        else:
            raise ValueError(f"shape mismatch: {value_shape} cannot "
                             f"be broadcast to {(None, ) + entry_shape}")
        return nrec

    @staticmethod
    def _chunks_autosize(max_trains, entry_shape, dtype, stype):
        """Caclulates chunk size"""
        MN = (max_trains, 32, 32)
        SZ = (1 << 14, 1 << 19, 1 << 23)  # 16K, 512K, 8M

        size = np.prod(entry_shape, dtype=int)
        ndim = len(entry_shape)
        nbytes = size * np.dtype(dtype).itemsize

        entry_type = int(size != 1) * (1 + int(ndim > 1))
        chunk = max(SZ[entry_type] // nbytes, MN[entry_type])
        if stype == 0:
            chunk = min(chunk, max_trains)

        return (chunk,) + tuple(entry_shape)

    def __call__(self, writer):
        return writer._ds_attrs[id(self)]

    def create(self, writer, grp):
        """Creates dataset in h5-file and return writer"""
        attr = self(writer)

        ds = grp.create_dataset(
            attr.key, (0,) + attr.entry_shape,
            dtype=attr.dtype, chunks=attr.chunks, maxshape=(None,)
            + attr.entry_shape, compression=attr.compression
        )
        if writer._meta.buffering:
            wrt = DatasetBufferedWriter(self, ds, attr.chunks)
        else:
            wrt = DatasetDirectWriter(self, ds, attr.chunks)

        return wrt

    def source(self, writer):
        raise NotImplementedError


class Dataset(DatasetBase):
    """Dataset descriptor"""
    source_class = Source

    def source(self, writer):
        attr = self(writer)
        return self.source_class(writer, attr.source_name, attr.stype)

    def get_mtrain_setter(self, name):
        """Returns suitable attribute setter instance"""
        return MultiTrainDataSetter(name)

    def get_strain_setter(self, name):
        """Returns suitable attribute setter instance"""
        return SingleTrainDataSetter(name)

    def get_dataset_fullname(self, writer):
        # expected to return (source_name, key, stype)
        source_name, key = self.canonical_name
        return Dataset._normalize_name(
            source_name.format(**writer.param), key)

    def get_chunks(self, writer):
        # expected to return chunks
        return Dataset._expand_shape(self.chunks, writer)

    def get_entry_attr(self, writer):
        # expected to return (entry_shape, dtype)
        return (Dataset._expand_shape(self.entry_shape, writer),
                self.dtype)

    @staticmethod
    def _normalize_name(source_name, key):
        """Transforms canonical name to the internal form"""
        # can we really distinguish sources by colons?
        stype = int(':' in source_name)
        if stype:
            tk, key = key.split('.', 1)
            source_name = source_name + '/' + tk

        return source_name, key.replace('.', '/'), stype

    @staticmethod
    def _expand_shape(shape_decl, writer):
        if isinstance(shape_decl, str):
            shape = tuple(writer.param[shape_decl])
        elif hasattr(shape_decl, '__iter__'):
            shape = tuple(writer.param[n] if isinstance(n, str) else n
                          for n in shape_decl)
        else:
            shape = shape_decl

        return shape
