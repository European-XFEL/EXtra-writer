# compatibility to future features
from . import future

import numpy as np

from itertools import accumulate


class DataQueue:
    def __init__(self):
        self.data = []
        self.size = []
        self.nwritten = 0
        self.nready = 0

    def __bool__(self):
        return bool(self.data)

    def append(self, count, data):
        """Appends data into queue for future writing"""
        ntrain = len(count)
        self.data.append((count, data))
        self.size.append((0, 0, ntrain))
        self.nready += ntrain

    def reset(self):
        """Reset counters for new file"""
        self.nready -= self.nwritten
        self.nwritten = 0

    def get_max_below(self, end):
        """Finds the maximum number of trains that data items in the queue
        fill without splitting and below the given number"""
        ntrain = self.nwritten
        for _, _, n in self.size:
            if ntrain + n > end:
                return ntrain
            ntrain += n
        return ntrain

    def write(self, writer, end):
        """Writes data items from the queue"""
        nrest = end - self.nwritten
        while nrest > 0:
            offset, train0, ntrain = self.size[0]
            if ntrain == 1 and offset == 0:
                count, data = self.data.pop(0)
                writer.write(data, count[0])
                nrest -= 1
                self.size.pop(0)
            elif ntrain <= nrest:
                count, data = self.data.pop(0)
                trainN = train0 + ntrain
                nrec = np.sum(count[train0:trainN])
                writer.write(data, nrec, offset)

                nrest -= ntrain
                self.size.pop(0)
            else:
                count, data = self.data[0]
                trainN = train0 + nrest
                nrec = np.sum(count[train0:trainN])
                writer.write(data, nrec, offset)

                self.size[0] = (offset + nrec, trainN, ntrain - nrest)
                nrest = 0

        self.nwritten = end


# Attention! Do not instanciate `Source` in the metaclass `FileWriterMeta`
class Source:
    """Creates data source group and its indexes"""

    SECTION = ('CONTROL', 'INSTRUMENT')

    def __init__(self, writer, name, stype=None):
        self.writer = writer
        self.name = name
        if stype is None:
            self.stype = int(':' in name)
        else:
            self.stype = stype

        self.section = self.SECTION[self.stype]
        if writer._meta.break_into_sequence:
            self.max_trains = writer._meta.max_train_per_file
        else:
            self.max_trains = None

        self.ndatasets = 0
        self.nready = 0

        self.datasets = []
        self.dsno = {}
        self.file_ds = []
        self.data = []

        self.count_buf = []

        self.first = []
        self.count = []
        self.nrec = 0

        self.block_writing = True

    def add(self, name, ds):
        """Adds dataset to the source"""
        self.dsno[name] = len(self.datasets)
        self.datasets.append(ds)
        self.data.append(DataQueue())
        self.file_ds.append(None)
        self.ndatasets += 1

    def create(self):
        """Creates all datasets in file"""
        grp = self.writer._file.create_group(self.section + '/' + self.name)
        for dsno, ds in enumerate(self.datasets):
            self.file_ds[dsno] = ds.create(self.writer, grp)
        self._grp = grp
        self.block_writing = False

        while self.nready >= self.ndatasets and not self.block_writing:
            self.write_data()

        return grp

    def create_index(self, index_grp, max_trains):
        """Create source index in h5-file"""
        grp = index_grp.create_group(self.name)
        for key in ('first', 'count'):
            ds = grp.create_dataset(
                key, (max_trains,), dtype=np.uint64, chunks=(max_trains,),
                maxshape=(None,)
            )
            ds[:] = 0

    def write_index(self, index_grp, ntrains):
        """Writes source index in h5-file"""
        nmissed = ntrains - len(self.count)
        if nmissed > 0:
            self.count += [0] * nmissed
            self.first += [self.nrec] * nmissed

        grp = index_grp[self.name]
        for dsname in ('first', 'count'):
            ds = grp[dsname]
            ds.resize(ntrains, axis=0)
            val = getattr(self, dsname)
            ds[:] = val[:ntrains]
            del val[:ntrains]

        if len(self.count):
            self.first = list(accumulate(self.count[:-1], initial=0))
            self.nrec = sum(self.count)
        else:
            self.first = []
            self.nrec = 0

        del self.count_buf[:ntrains]

    def close_datasets(self):
        """Finalize writing"""
        for dsno, ds in enumerate(self.datasets):
            self.file_ds[dsno].flush()
            self.data[dsno].reset()

    def add_train_data(self, nrec, name, value):
        """Adds single train data to the source"""
        if self.stype == 0 and nrec != 1:
            raise ValueError("maximum one entry per train can be written "
                             "in control source")

        dsno = self.dsno[name]

        ntrain = self.data[dsno].nready
        nwritten = len(self.count)
        if nwritten > ntrain:
            if self.count_buf[ntrain] != nrec:
                raise ValueError("count mismatch the number of frames "
                                 "in source")
        else:
            self.count_buf.append(nrec)

        self._put_data(dsno, [nrec], value)

    def add_data(self, count, name, value):
        """Adds multitrain data to the source"""
        if self.stype == 0 and np.any(count > 1):
            raise ValueError("maximum one entry per train can be written "
                             "in control source")

        dsno = self.dsno[name]

        ntrain = self.data[dsno].nready
        nwritten = len(self.count)
        if nwritten > ntrain:
            nmatch = min(nwritten - ntrain, len(count))
            if np.any(self.count_buf[ntrain:ntrain+nmatch] != count[:nmatch]):
                raise ValueError("count mismatch the number of frames "
                                 "in source")

            self.count_buf += count[nmatch:].tolist()
        else:
            self.count_buf += count.tolist()

        self._put_data(dsno, count, value)

    def _put_data(self, dsno, count, value):
        self.nready += not self.data[dsno]
        self.data[dsno].append(count, value)

        while self.nready >= self.ndatasets and not self.block_writing:
            self.write_data()

    def write_data(self):
        """Write data when the trains completely filled"""
        train0 = len(self.count)
        max_ready = min(d.nready for d in self.data)

        if self.max_trains is not None and self.max_trains < max_ready:
            max_ready = self.max_trains
            self.block_writing = True

        self.nready = 0
        trainN = train0
        for dsno in range(self.ndatasets):
            if self.block_writing:
                end = max_ready
            else:
                end = self.data[dsno].get_max_below(max_ready)

            self.data[dsno].write(self.file_ds[dsno], end)
            self.nready += bool(self.data[dsno])
            trainN = max(trainN, end)

        count = self.count_buf[train0:trainN]
        first = list(accumulate(count[:-1], initial=self.nrec))
        self.count += count
        self.first += first
        self.nrec += np.sum(count, dtype=int)

    def get_ntrain(self, dsname):
        dsno = self.dsno[dsname]
        return self.data[dsno].nready

    def get_min_trains(self):
        return min(d.nready for d in self.data)
