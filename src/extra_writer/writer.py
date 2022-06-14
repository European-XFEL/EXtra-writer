import h5py
import numpy as np

from .dataset_descriptor import DatasetBase, Dataset, BlockedSetter


class Options:
    """Provides a set of options with overriding default values
    by ones declared in Meta subclass
    """
    NAMES = (
        'max_train_per_file', 'break_into_sequence',
        'class_attrs_interface', 'buffering', 'aliases'
    )

    def __init__(self, meta=None, base=None):
        self.max_train_per_file = 500
        self.break_into_sequence = False
        self.warn_on_missing_data = False
        self.class_attrs_interface = True
        self.buffering = True
        self.aliases = {}

        self.copy(base)
        self.override_defaults(meta)

    def copy(self, opts):
        if not opts:
            return
        for attr_name in Options.NAMES:
            setattr(self, attr_name, getattr(opts, attr_name))

    def override_defaults(self, meta):
        if not meta:
            return
        meta_attrs = meta.__dict__.copy()
        for attr_name in meta.__dict__:
            if attr_name.startswith('_'):
                del meta_attrs[attr_name]

        for attr_name in Options.NAMES:
            if attr_name in meta_attrs:
                val = meta_attrs.pop(attr_name)
                if isinstance(val, dict):
                    getattr(self, attr_name).update(val)
                else:
                    setattr(self, attr_name, val)

        if meta_attrs != {}:
            raise TypeError("'class Meta' got invalid attribute(s): " +
                            ','.join(meta_attrs))


class FileWriterMeta(type):
    """Constructs writer class"""
    def __new__(cls, name, bases, attrs):
        attr_meta = attrs.pop('Meta', None)

        new_attrs = {}
        datasets = {}
        for base in reversed(bases):
            if issubclass(base, FileWriterBase):
                datasets.update(base.datasets)

        for key, val in attrs.items():
            if isinstance(val, DatasetBase):
                datasets[key] = val
            else:
                new_attrs[key] = val

        new_attrs['datasets'] = datasets
        new_class = super().__new__(cls, name, bases, new_attrs)

        meta = attr_meta or getattr(new_class, 'Meta', None)
        base_meta = getattr(new_class, '_meta', None)
        new_class._meta = Options(meta, base_meta)

        for ds_name, ds in datasets.items():
            ds.resolve_name(new_class._meta.aliases)
            if new_class._meta.class_attrs_interface:
                setattr(new_class, ds_name, ds.get_attribute_setter(ds_name))
            else:
                setattr(new_class, ds_name, BlockedSetter())

        return new_class


class FileWriterBase(object):
    """Writes data in EuXFEL format"""
    datasets = {}

    def __new__(cls, *args, **kwargs):
        if not cls.datasets:
            raise TypeError(f"Can't instantiate class {cls.__name__}, "
                            "because it has no datasets")
        return super().__new__(cls)

    def __init__(self, filename, **kwargs):
        self._ds_attrs = {}
        self._train_data = {}
        self.trains = []
        self.timestamp = []
        self.flags = []
        self.seq = 0
        self.filename = filename
        self.param = kwargs

        source_decl = {}
        for ds_name, ds in self.datasets.items():
            ds.init_dataset_attr(self)
            ds_attr = ds(self)
            repr_ds = source_decl.setdefault(ds_attr.source_name, ds)
            if issubclass(ds.source_class, repr_ds.source_class):
                source_decl[ds_attr.source_name] = ds
            elif not issubclass(repr_ds.source_class, ds.source_class):
                raise TypeError("Incompatible datasets in the source "
                                f"'{ds_attr.source_name}'")

        self.nsource = len(source_decl)
        self.sources = {}
        self.list_of_sources = []
        self.source_ntrain = {}
        for src_name, ds in source_decl.items():
            src = ds.source(self)
            self.sources[src_name] = src
            self.source_ntrain[src_name] = 0
            self.list_of_sources.append((src.section, src_name))

        for dsname, ds in self.datasets.items():
            src_name = ds(self).source_name
            self.sources[src_name].add(dsname, ds)

        file = h5py.File(filename.format(seq=self.seq), 'w')
        try:
            self.init_file(file)
        except Exception as e:
            file.close()
            raise e

    def init_file(self, file):
        """Initialises a new file"""
        self._file = file
        self.write_metadata()
        self.create_indices()
        self.create_datasets()

    def close(self):
        """Finalises writing and close a file"""
        self.rotate_sequence_file(True)
        self.close_datasets()
        self.write_indices()
        self._file.close()

    def write_metadata(self):
        """Write the METADATA section, including lists of sources"""
        from . import __version__
        vlen_bytes = h5py.special_dtype(vlen=bytes)  # HDF5 vlen string, ASCII

        meta_grp = self._file.create_group('METADATA')
        meta_grp.create_dataset('dataFormatVersion', dtype=vlen_bytes,
                                data=['1.0'])
        meta_grp.create_dataset('daqLibrary', dtype=vlen_bytes,
                                data=[f'EXtra-data {__version__}'])
        # TODO?: creationDate, karaboFramework, proposalNumber, runNumber,
        #  sequenceNumber, updateDate

        sources_grp = meta_grp.create_group('dataSources')
        sources_grp.create_dataset('dataSourceId', dtype=vlen_bytes, data=[
            sect + '/' + src for sect, src in self.list_of_sources
        ])

        sections, sources = (zip(*self.list_of_sources)
                             if self.nsource else (None, None))
        sources_grp.create_dataset('root', dtype=vlen_bytes, data=sections)
        sources_grp.create_dataset('deviceId', dtype=vlen_bytes, data=sources)

    def create_indices(self):
        """Creates and allocate the datasets for indices in the file
        but doesn't write real data"""
        max_trains = self._meta.max_train_per_file
        index_datasets = [
            ('trainId', np.uint64),
            ('timestamp', np.uint64),
            ('flag', np.uint32),
        ]
        self.index_grp = self._file.create_group('INDEX')
        for key, dtype in index_datasets:
            ds = self.index_grp.create_dataset(
                key, (max_trains,), dtype=dtype, chunks=(max_trains,),
                maxshape=(None,)
            )
            ds[:] = 0

        for sname, src in self.sources.items():
            src.create_index(self.index_grp, max_trains)

    def write_indices(self):
        """Write real indices to the file"""
        ntrains = len(self.trains)
        if self._meta.break_into_sequence:
            ntrains = min(self._meta.max_train_per_file, ntrains)
        index_datasets = [
            ('trainId', self.trains),
            ('timestamp', self.timestamp),
            ('flag', self.flags),
        ]
        for key, data in index_datasets:
            ds = self.index_grp[key]
            ds.resize(ntrains, 0)
            ds[:] = data[:ntrains]

        for src_name, src in self.sources.items():
            src.write_index(self.index_grp, ntrains)
            self.source_ntrain[src_name] = src.get_min_trains()

        del self.trains[:ntrains]
        del self.timestamp[:ntrains]
        del self.flags[:ntrains]

    def create_datasets(self):
        """Creates datasets in the file"""
        for sname, src in self.sources.items():
            src.create()

    def close_datasets(self):
        """Writes rest of buffered data in datasets and set final size"""
        for src in self.sources.values():
            src.close_datasets()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def add_value(self, count, name, value):
        """Fills a single dataset across multiple trains"""
        ds = self.datasets.get(name)
        if ds is None:
            raise KeyError(f"'{type(self).__qualname__}' does not have "
                           f"a dataset named '{name}'")

        # check shape value
        nrec = ds.check_value(self, value)
        ntrain = np.size(count)
        if nrec != np.sum(count):
            raise ValueError("total counts is not equal to "
                             "the number of records")

        # check count
        src_name = ds(self).source_name
        src = self.sources[src_name]
        if src.get_ntrain(name) + ntrain > len(self.trains):
            raise ValueError("the number of trains in this data exeeds "
                             "the number of trains in file")

        src.add_data(np.array(count), name, value)

        self.source_ntrain[src_name] = src.get_min_trains()
        self.rotate_sequence_file()

    def add_train_value(self, name, value):
        """Fills a single dataset in the current train"""
        ds = self.datasets.get(name)
        if ds is None:
            raise KeyError(f"'{type(self).__qualname__}' does not have "
                           f"a dataset named '{name}'")

        nrec = ds.check_value(self, value)

        # check count
        src_name = ds(self).source_name
        src = self.sources[src_name]
        if src.get_ntrain(name) + 1 > len(self.trains):
            raise ValueError("the number of trains in this data exeeds "
                             "the number of trains in file")

        src.add_train_data(nrec, name, value)

        self.source_ntrain[src_name] = src.get_min_trains()
        self.rotate_sequence_file()

    def add_data(self, count, **kwargs):
        """Adds data"""
        for name, value in kwargs.items():
            self.add_value(count, name, value)

    def add_train_data(self, **kwargs):
        """Adds data to the current train"""
        for name, value in kwargs.items():
            self.add_train_value(name, value)

    def rotate_sequence_file(self, finalize=False):
        """opens a new sequence file if necessary"""
        if self._meta.break_into_sequence and self.nsource:
            op = (lambda a: max(a)) if finalize else (lambda a: min(a))
            ntrain = op(self.source_ntrain.values())
            while ntrain > self._meta.max_train_per_file:

                self.close_datasets()
                self.write_indices()
                self._file.close()

                self.seq += 1

                file = h5py.File(self.filename.format(seq=self.seq), 'w')
                self.init_file(file)

                ntrain = op(self.source_ntrain.values())

    def add_trains(self, tid, ts):
        """Adds trains to the file"""
        ntrain = len(tid)
        if ntrain != len(ts):
            raise ValueError("arguments must have the same size")

        self.trains += tid
        self.timestamp += ts
        self.flags += [1] * ntrain

    def add_train(self, tid, ts):
        self.trains.append(tid)
        self.timestamp.append(ts)
        self.flags.append(1)


class FileWriter(FileWriterBase, metaclass=FileWriterMeta):
    """Writes data into European XFEL file format

    Create a new class inherited from :class:`FileWriter`
    and use :class:`DS` to declare datasets:

    .. code-block:: python

        from extra_writer import FileWriter, DS, trs_

        class MyFileWriter(FileWriter):
            gv = DS('@ctrl', 'geom.fragmentVectors', ('nfrag',3,3), float)
            nb = DS('@ctrl', 'param.numberOfBins', (), np.uint64)

            tid = DS('@inst', 'azimuthal.trainId', (), np.uint64)
            pid = DS('@inst', 'azimuthal.pulseId', (), np.uint64)
            v = DS('@inst', 'azimuthal.profile', ('nbin',), float)

            class Meta:
                max_train_per_file = 40
                break_into_sequence = True
                aliases = {
                    'ctrl': '{det_name}/x/y',
                    'inst': '{det_name}/x/y:output'
                }

    Subclass :class:`Meta` is a special class for options.

    Use new class to write data in files by trains:

    .. code-block:: python

        nbin, nfrag = 1000, 4
        det_name = 'MID_DET_AGIPD1M-1'
        gv = np.zeros([nfrag, 3, 3])

        pulses = list(range(0, 400, 4))
        npulse = len(pulses)
        trains = list(range(100001, 100101))
        ntrain = len(trains)

        prm = {'det_name': det_name, 'nbin': nbin, 'nfrag': nfrag}
        filename = 'mydata-{seq:03d}.h5'
        with MyFileWriter(filename, **prm) as wr:

            for tid in trains:
                # create/compute data
                v = np.random.rand(npulse, nbin)
                # add train
                wr.add_train(tid, 0)
                # add data (funcion kwargs interface)
                wr.add_train_data(gv=gv, nb=nbin)
                # add data (class attribute interface)
                wr.tid = [tid] * npulse
                wr.pid = pulses
                wr.v = v

    Multi-train arrays can also be written at once:

    .. code-block:: python

        with MyFileWriter(filename, **prm) as wr:
            # create/compute data
            v = np.random.rand(ntrain*npulse, nbin)
            tid, pid = np.meshgrid(trains, pulses, indexing='ij')
            nrec = [npulse] * ntrain
            # add train
            wr.add_trains(trains, [0] * ntrain)
            # add data (funcion kwargs interface)
            wr.add_data(
                [1]*ntrain,
                gv=gv[None,...].repeat(ntrain, 0),
                nb=[nbin]*ntrain,
            )
            # add data (class attribute interface)
            wr.tid = trs_(nrec, tid.flatten())
            wr.pid = trs_(nrec, pid.flatten())
            wr.v = trs_(nrec, v)

    Writing of multi-train arrays and writing by trains may be combined.
    Remember that only those trains are written to the file for which all
    the keys of one source are filled with data.

    For the sources in 'CONTROL' section, only one entry is allowed per train.

    For the sources in 'INSTRUMENT' section, the number of entries may vary
    from train to train. All datasets in one source must have the same number
    of entries in the same train.
    """
    @classmethod
    def open(cls, fn, datasets, **kwargs):
        """Constructs and instantiates a child class inheriting `FileWriter`.
        
        Constructs a child class inheriting :class:`FileWriter`. Attributes for
        the new class should be given in the `dataset` dict. Keys in the dict must
        be the identifiers and values must inherit from descripotor :class:Dataset.
        
        Other keyword arguments are transformed into attrubutes of special nested
        class `Meta`.

        Example:

        .. code-block:: python
            datasets = dict(
                gv = DS('@ctrl', 'geom.fragmentVectors', (4,3,3), float),
                nb = DS('@ctrl', 'param.numberOfBins', (), np.uint64),
            )
            aliases = {
                'ctrl': 'MID_DET_AGIPD1M-1/x/y',
            }
            filename = 'mydata-{seq:03d}.h5'
            wr = FileWriter.open(filename, datasets, aliases=aliases,
                                 break_into_sequence=True)
        
        this is equivalent to
        
        .. code-block:: python
            class MyFileWriter(FileWriter):
                gv = DS('@ctrl', 'geom.fragmentVectors', (4,3,3), float),
                nb = DS('@ctrl', 'param.numberOfBins', (), np.uint64),
        
                class Meta:
                    break_into_sequence = True
                    aliases = {
                        'ctrl': '{det_name}/x/y',
                    }

            wr = MyFileWriter('mydata-{seq:03d}.h5')

        There is no expansion of substitutes and parameters in this constructor
        since the constructed class is not reused.
        """
        class_name = cls.__name__ + '_' + str(id(datasets))

        aliases = kwargs.get('aliases', {})
        attrs = {}
        for name, val in datasets.items():
            if isinstance(val, dict):
                for ds_name, ds in val.items():
                    if isinstance(ds, Dataset):
                        isalias, src_id, key = ds.orig_name
                        src_suffix = aliases[src_id] if isalias else src_id
                        new_name = (name + '/' + src_suffix
                                    if src_suffix else name)
                        ds.set_name(new_name, key)
                    attrs[ds_name] = ds
            else:
                attrs[name] = val

        if kwargs:
            attrs['Meta'] = type(class_name + '.Meta', (object,), kwargs)

        newcls = type(class_name, (cls,), attrs)
        return newcls(fn)
