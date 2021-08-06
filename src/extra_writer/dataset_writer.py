import numpy as np


class DatasetWriterBase:
    """Abstract class for writers wrapping h5py"""
    def __init__(self, ds, file_ds, chunks):
        self.ds = ds
        self.file_ds = file_ds
        self.chunks = chunks
        self.pos = 0

    def flush(self):
        pass

    def write(self, data, nrec, start=None):
        raise NotImplementedError


class DatasetDirectWriter(DatasetWriterBase):
    """Class writes data directly to the file"""
    def write(self, data, nrec, start=None):
        """Writes data to disk"""
        end = self.pos + nrec
        if start is None:
            start = 0
        self.file_ds.resize(end, 0)
        if not np.ndim(data):
            self.file_ds[self.pos:end] = data
        else:
            # self.file_ds[self.pos:end] = data[start:start+nrec]
            self.file_ds.write_direct(
                data, np.s_[start:start+nrec], np.s_[self.pos:end])


class DatasetBufferedWriter(DatasetWriterBase):
    """Class implements buffered writing"""
    def __init__(self, ds, file_ds, chunks):
        super().__init__(ds, file_ds, chunks)
        self._data = np.empty(chunks, dtype=ds.dtype)
        self.size = chunks[0]
        self.nbuf = 0

    def flush(self):
        """Writes buffer to disk"""
        if self.nbuf:
            end = self.pos + self.nbuf
            self.file_ds.resize(end, 0)
            self.file_ds.write_direct(
                self._data, np.s_[:self.nbuf], np.s_[self.pos:end])
            self.pos = end
            self.nbuf = 0

    def write_one(self, value):
        """Buffers single record"""
        self._data[self.nbuf] = value
        self.nbuf += 1
        if self.nbuf >= self.size:
            self.flush()

    def write_many(self, arr, nrec, start=None):
        """Buffers multiple records"""
        if start is None:
            start = 0
        buf_nrest = self.size - self.nbuf
        data_nrest = nrec - buf_nrest
        if data_nrest < 0:
            # copy
            end = self.nbuf + nrec
            self._data[self.nbuf:end] = arr[start:start+nrec]
            self.nbuf = end
        elif self.nbuf and data_nrest < self.size:
            # copy, flush, copy
            split = start + buf_nrest
            self._data[self.nbuf:] = arr[start:split]

            end = self.pos + self.size
            self.file_ds.resize(end, 0)
            self.file_ds.write_direct(
                self._data, np.s_[:], np.s_[self.pos:end])
            self.pos = end

            self._data[:data_nrest] = arr[split:start+nrec]
            self.nbuf = data_nrest
        else:
            # flush, write, copy
            nrest = nrec % self.size
            nwrite = nrec - nrest

            split = self.pos + self.nbuf
            end = split + nwrite
            self.file_ds.resize(end, 0)
            if self.nbuf:
                self.file_ds.write_direct(
                    self._data, np.s_[:self.nbuf], np.s_[self.pos:split])
            self.file_ds.write_direct(arr, np.s_[start:start+nwrite],
                                      np.s_[split:end])

            self._data[:nrest] = arr[start+nwrite:start+nrec]
            self.pos = end
            self.nbuf = nrest

    def write(self, data, nrec, start=None):
        """Buffer or writes data"""
        if nrec == 1:
            self.write_one(data[start] if start is not None else data)
        else:
            if not isinstance(data, np.ndarray):
                arr = np.array(data, dtype=self.ds.dtype)
            else:
                arr = data

            # arr = np.broadcast_to(data, (nrec,) + self.ds.entry_shape)
            self.write_many(arr, nrec, start=start)
