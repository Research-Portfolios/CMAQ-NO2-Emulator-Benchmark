"""A minimal reader of netCDF classic files (CDF-1, CDF-2 and CDF-5) through any seekable file object, with the small part of the h5py interface that the reducers use, so that classic-format members can be read where netCDF4 is absent."""
import struct
import numpy as np
NC_DIMENSION, NC_VARIABLE, NC_ATTRIBUTE = (10, 11, 12)
DT = {1: 'i1', 2: 'S1', 3: '>i2', 4: '>i4', 5: '>f4', 6: '>f8', 7: 'u1', 8: '>u2', 9: '>u4', 10: '>i8', 11: '>u8'}

class ClassicError(ValueError):
    pass

class _Buf:

    def __init__(self, f):
        self.f = f
        self.f.seek(0)
        self.data = b''
        self.pos = 0

    def take(self, n):
        while self.pos + n > len(self.data):
            more = self.f.read(max(65536, n))
            if not more:
                raise ClassicError('header ends early')
            self.data += more
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

class ClassicVar:

    def __init__(self, cf, name, dims, attrs, nctype, vsize, begin):
        self.cf = cf
        self.name = name
        self.dim_names = [cf.dim_names[i] for i in dims]
        self.is_record = bool(dims) and dims[0] == cf.rec_dim
        self.attrs = attrs
        self.nctype = nctype
        self.raw_dtype = np.dtype(DT[nctype])
        self.dtype = self.raw_dtype.newbyteorder('=')
        self.vsize = vsize
        self.begin = begin
        self.chunks = None
        shape = [cf.dim_lens[i] for i in dims]
        if self.is_record:
            shape[0] = cf.numrecs
        self.shape = tuple(shape)
        self.dimensions = tuple(self.dim_names)

    def _per_record(self):
        return int(np.prod(self.shape[1:], dtype=np.int64)) * self.raw_dtype.itemsize

    def _read(self, off, n):
        self.cf.f.seek(off)
        b = self.cf.f.read(n)
        if len(b) != n:
            raise ClassicError('%s: short read' % self.name)
        return b

    def _records(self, recs):
        per = self._per_record()
        out = np.empty((len(recs),) + self.shape[1:], dtype=self.dtype)
        for k, r in enumerate(recs):
            b = self._read(self.begin + r * self.cf.recsize, per)
            out[k] = np.frombuffer(b, dtype=self.raw_dtype).reshape(self.shape[1:]).astype(self.dtype)
        return out

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        if idx == (Ellipsis,) or idx == (slice(None),):
            idx = ()
        if not self.is_record:
            n = int(np.prod(self.shape, dtype=np.int64)) * self.raw_dtype.itemsize
            a = np.frombuffer(self._read(self.begin, n), dtype=self.raw_dtype).reshape(self.shape)
            return np.array(a.astype(self.dtype)[idx] if idx else a.astype(self.dtype))
        first = idx[0] if idx else slice(None)
        rest = idx[1:] if idx else ()
        if isinstance(first, (int, np.integer)):
            r = int(first)
            if r < 0:
                r += self.shape[0]
            if not 0 <= r < self.shape[0]:
                raise IndexError('record %d of %d' % (r, self.shape[0]))
            a = self._records([r])[0]
            return np.array(a[rest] if rest else a)
        if not isinstance(first, slice):
            raise TypeError('record index must be an int or a slice')
        recs = list(range(*first.indices(self.shape[0])))
        a = self._records(recs)
        return np.array(a[(slice(None),) + tuple(rest)] if rest else a)

class ClassicFile:

    def __init__(self, f):
        self.f = f
        b = _Buf(f)
        magic = b.take(4)
        if magic[:3] != b'CDF' or magic[3] not in (1, 2, 5):
            raise ClassicError('not a netCDF classic file')
        self.version = magic[3]
        cnt = '>Q' if self.version == 5 else '>I'
        csz = 8 if self.version == 5 else 4
        osz = 4 if self.version == 1 else 8

        def count():
            return struct.unpack(cnt, b.take(csz))[0]

        def name():
            n = count()
            s = b.take(n)
            b.take((4 - n % 4) % 4)
            return s.decode('utf-8', 'replace')

        def attrs():
            tag = struct.unpack('>I', b.take(4))[0]
            n = count()
            out = {}
            if tag == 0:
                return out
            if tag != NC_ATTRIBUTE:
                raise ClassicError('bad attribute list tag %d' % tag)
            for _ in range(n):
                nm = name()
                t = struct.unpack('>I', b.take(4))[0]
                k = count()
                dt = np.dtype(DT[t])
                raw = b.take(k * dt.itemsize)
                b.take((4 - k * dt.itemsize % 4) % 4)
                if t == 2:
                    out[nm] = raw.decode('latin-1').rstrip('\x00 ')
                else:
                    out[nm] = np.frombuffer(raw, dtype=dt).astype(dt.newbyteorder('='))
            return out
        nr = b.take(csz)
        self.numrecs = struct.unpack(cnt, nr)[0]
        if self.numrecs in (4294967295, 18446744073709551615):
            raise ClassicError('streaming numrecs is not supported')
        tag = struct.unpack('>I', b.take(4))[0]
        nd = count()
        self.dim_names, self.dim_lens, self.rec_dim = ([], [], None)
        if tag not in (0, NC_DIMENSION):
            raise ClassicError('bad dimension list tag %d' % tag)
        for i in range(nd if tag else 0):
            self.dim_names.append(name())
            ln = count()
            self.dim_lens.append(ln)
            if ln == 0:
                self.rec_dim = i
        self.attrs = attrs()
        tag = struct.unpack('>I', b.take(4))[0]
        nv = count()
        if tag not in (0, NC_VARIABLE):
            raise ClassicError('bad variable list tag %d' % tag)
        raw = []
        for _ in range(nv if tag else 0):
            nm = name()
            k = count()
            dims = [count() for _ in range(k)]
            va = attrs()
            t = struct.unpack('>I', b.take(4))[0]
            vsize = count()
            begin = struct.unpack('>I' if osz == 4 else '>Q', b.take(osz))[0]
            raw.append((nm, dims, va, t, vsize, begin))
        recs = [r for r in raw if r[1] and r[1][0] == self.rec_dim]
        self.recsize = sum((r[4] for r in recs))
        if len(recs) == 1 and np.dtype(DT[recs[0][3]]).itemsize < 4:
            shp = [self.dim_lens[i] for i in recs[0][1][1:]]
            self.recsize = int(np.prod(shp, dtype=np.int64)) * np.dtype(DT[recs[0][3]]).itemsize
        self.variables = {}
        for nm, dims, va, t, vsize, begin in raw:
            self.variables[nm] = ClassicVar(self, nm, dims, va, t, vsize, begin)

    def __contains__(self, name):
        return name in self.variables

    def __getitem__(self, name):
        return self.variables[name]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
