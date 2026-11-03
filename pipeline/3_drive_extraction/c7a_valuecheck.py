"""Checks the extracted values with a separate decoder that shares no code with the extraction: for a fixed sample of members it walks the tar, decodes the raw HDF5 chunks itself, recomputes the daily means and cell sums, and compares every array bit for bit with the extract."""
import argparse
import datetime
import hashlib
import io
import json
import os
import re
import sys
import textwrap
import time
import zlib
import numpy as np
MERGED_MMDD = ('0115', '0415', '0701', '1015')
DAILY = ('ptegu', 'cmv_c1c2_12', 'cmv_c3_12')
DAILY_MMDD = '0701'
REP = ('ptnonipm', 'pt_oilgas', 'othpt')
REP_MONTH = '07'
NH = 24
BASE = 1
SPECIES_M = ('NO', 'NO2', 'HONO', 'VOC_INV')
SPECIES_P = ('NO', 'NO2', 'HONO')

class Unsupported(Exception):
    pass

def octal(f):
    if f and f[0] & 128:
        return int.from_bytes(f[1:], 'big')
    s = f.split(b'\x00', 1)[0].strip()
    return int(s, 8) if s else 0

def walk(path):
    size = os.path.getsize(path)
    out, off, longname, pax = ([], 0, None, None)
    with open(path, 'rb') as fh:
        while off + 512 <= size:
            fh.seek(off)
            h = fh.read(512)
            if h == bytes(512):
                break
            sz = octal(h[124:136])
            uns = sum(h[:148]) + 256 + sum(h[156:])
            sig = sum((b - 256 if b > 127 else b for b in h[:148] + h[156:])) + 256
            ok = octal(h[148:156]) in (uns, sig)
            typ = h[156:157]
            name = h[0:100].split(b'\x00', 1)[0]
            if h[257:263] == b'ustar\x00':
                pre = h[345:500].split(b'\x00', 1)[0]
                if pre:
                    name = pre + b'/' + name
            data = off + 512
            step = data + (sz + 511) // 512 * 512
            if typ == b'L':
                fh.seek(data)
                longname = fh.read(sz).split(b'\x00', 1)[0]
                off = step
                continue
            if typ in (b'x', b'g'):
                fh.seek(data)
                rec = fh.read(sz)
                for m in re.finditer(b'\\d+ path=([^\\n]*)\\n', rec):
                    if typ == b'x':
                        pax = m.group(1)
                off = step
                continue
            if longname is not None:
                name = longname
            if pax is not None:
                name = pax
            out.append((name.decode('utf-8', 'replace'), sz, data, ok, typ))
            longname = pax = None
            off = step if typ in (b'0', b'\x00', b'7') else data
    return out

class Win(io.RawIOBase):

    def __init__(self, fh, start, size):
        super().__init__()
        self.fh, self.start, self.size, self.pos = (fh, start, size, 0)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), max(0, self.size - self.pos))
        if n <= 0:
            return 0
        self.fh.seek(self.start + self.pos)
        d = self.fh.read(n)
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)

def unshuffle(buf, size):
    n = len(buf) // size
    body = np.frombuffer(buf[:n * size], dtype=np.uint8)
    out = body.reshape(size, n).T.reshape(-1).tobytes()
    return out + buf[n * size:]

def unfilter(raw, filters, mask, itemsize):
    buf = raw
    for i in reversed(range(len(filters))):
        if mask & 1 << i:
            continue
        code = filters[i][0]
        if code == 1:
            buf = zlib.decompress(buf)
        elif code == 2:
            buf = unshuffle(buf, itemsize)
        elif code == 3:
            buf = buf[:-4]
        else:
            raise Unsupported('filter %d' % code)
    return buf

def decode(fh, data_off, ds, nrec):
    import h5py
    dt = np.dtype(ds.dtype.str)
    shape = ds.shape
    dcpl = ds.id.get_create_plist()
    filters = [dcpl.get_filter(i) for i in range(dcpl.get_nfilters())]
    lay = dcpl.get_layout()
    nrec = min(nrec, shape[0])
    if lay == h5py.h5d.CONTIGUOUS:
        if filters:
            raise Unsupported('contiguous with filters')
        off = ds.id.get_offset()
        per = int(np.prod(shape[1:], dtype=np.int64)) * dt.itemsize
        fh.seek(data_off + off)
        raw = fh.read(per * nrec)
        return np.frombuffer(raw, dtype=dt).reshape((nrec,) + shape[1:])
    if lay != h5py.h5d.CHUNKED:
        raise Unsupported('layout %s' % lay)
    ch = ds.chunks
    out = np.empty((nrec,) + shape[1:], dtype=dt)
    ranges = [range(0, nrec, ch[0])] + [range(0, shape[k], ch[k]) for k in range(1, len(shape))]
    for org in np.ndindex(*[len(r) for r in ranges]):
        o = tuple((ranges[k][org[k]] for k in range(len(shape))))
        ci = ds.id.get_chunk_info_by_coord(o)
        if ci.byte_offset is None:
            raise Unsupported('an unallocated chunk')
        fh.seek(data_off + int(ci.byte_offset))
        raw = fh.read(int(ci.size))
        buf = unfilter(raw, filters, int(ci.filter_mask), dt.itemsize)
        block = np.frombuffer(buf, dtype=dt)[:int(np.prod(ch))].reshape(ch)
        ends = [min(o[k] + ch[k], nrec if k == 0 else shape[k]) for k in range(len(shape))]
        dst = tuple((slice(o[k], ends[k]) for k in range(len(shape))))
        src = tuple((slice(0, ends[k] - o[k]) for k in range(len(shape))))
        out[dst] = block[src]
    return out

def read_vars(fh, data_off, size, ext, names):
    out, units = ({}, {})
    if ext == 'ncf':
        from scipy.io import netcdf_file
        fh.seek(data_off)
        bio = io.BytesIO(fh.read(size))
        f = netcdf_file(bio, 'r', mmap=False, maskandscale=False)
        try:
            for n in names:
                if n not in f.variables:
                    continue
                v = f.variables[n]
                a = np.array(v.data)
                out[n] = a[:NH] if a.shape[0] > 1 else a
                u = getattr(v, 'units', b'')
                units[n] = (u.decode('latin-1') if isinstance(u, bytes) else str(u)).strip('\x00 ')
        finally:
            f.close()
        return (out, units)
    import h5py
    with h5py.File(Win(fh, data_off, size), 'r') as f:
        for n in names:
            key = '_nc4_non_coord_' + n
            if key not in f:
                key = n
            if key not in f:
                continue
            ds = f[key]
            out[n] = decode(fh, data_off, ds, NH)
            u = ds.attrs.get('units')
            if isinstance(u, bytes):
                u = u.decode('latin-1')
            elif isinstance(u, np.ndarray):
                u = b''.join(u.ravel().tolist()).decode('latin-1') if u.dtype.kind == 'S' else str(u.ravel()[0])
            units[n] = str(u).strip('\x00 ')
    return (out, units)

def whole_sha(fh, off, size):
    h = hashlib.sha256()
    fh.seek(off)
    left = size
    while left:
        d = fh.read(min(left, 8 << 20))
        if not d:
            raise IOError('short read')
        h.update(d)
        left -= len(d)
    return h.hexdigest()

def means(a):
    return a.astype(np.float64).sum(axis=0) / 24.0

def yyyyddd(d):
    x = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
    return x.year * 1000 + x.timetuple().tm_yday

def tflag_ok(tf, d):
    want_d = yyyyddd(d)
    t = np.asarray(tf)[:NH]
    return bool((t[:, :, 0] == want_d).all() and (t[:, :, 1] == (np.arange(NH) * 10000)[:, None]).all())

def ndiff(a, b):
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return -1
    return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))

def find(year, h_input, tar_re, mem_re):
    folder = os.path.join(h_input, year, 'emis')
    for name in sorted(os.listdir(folder)):
        if re.search(tar_re, name) and name.endswith('.tar'):
            path = os.path.join(folder, name)
            hits = [m for m in walk(path) if re.search(mem_re, m[0])]
            return (name, path, hits)
    return (None, None, [])

def pick(hits):
    nc4 = [h for h in hits if h[0].endswith('.nc4')]
    return (nc4 or hits or [None])[0]

def check_member(year, kind, sec, d, h_input, ydir, grid, groups):
    rec = {'kind': kind, 'sector': sec, 'date': d}
    mm = d[4:6] if d else REP_MONTH
    if kind == 'merged':
        tre = '_%s_merged_nobeis_norwc_%s_EQUATES_v1\\.0\\.tar$' % (year, mm)
        mre = 'emis_mole_all_%s_12US1_nobeis_norwc_' % d
        epath = os.path.join(ydir, 'emis_merged', 'emis_merged_%s.npz' % d)
        names = SPECIES_M + ('TFLAG',)
    else:
        tre = 'ptsectors_plus_rwc_%s_%s_EQUATES_v1\\.0\\.tar$' % (year, mm)
        mre = 'inln_mole_%s_%s_12US1_' % (sec, d if d else year + REP_MONTH + '\\d\\d')
        names = SPECIES_P + ('TFLAG',)
    tar, tpath, hits = find(year, h_input, tre, mre)
    if kind == 'pt' and (not d):
        hits = sorted(hits, key=lambda h: re.search('_(\\d{8})_12US1_', h[0]).group(1))
        hits = [h for h in hits if h[0].endswith('.nc4')] or hits
        if hits:
            d = re.search('_(\\d{8})_12US1_', hits[0][0]).group(1)
            rec['date'] = d
    m = pick(hits)
    if m is None:
        rec['found'] = False
        return rec
    if kind == 'pt':
        epath = os.path.join(ydir, 'pt_marine', sec, 'pt_%s_%s.npz' % (sec, d))
    name, size, off, hok, _ = m
    rec.update(found=True, tar=tar, member=name, header_checksum_ok=hok)
    if not os.path.exists(epath):
        rec['extract'] = 'NOT FOUND'
        return rec
    with np.load(epath, allow_pickle=False) as zf:
        z = {k: zf[k] for k in zf.files}
    with open(epath, 'rb') as fe:
        rec['extract_sha256'] = hashlib.sha256(fe.read()).hexdigest()
    rec['meta_equal'] = bool(str(z['tar']) == tar and str(z['member']) == name and (int(z['member_offset']) == off) and (int(z['member_size']) == size))
    ext = name.rsplit('.', 1)[-1]
    t0 = time.time()
    fh = open(tpath, 'rb')

    def one(n):
        return read_vars(fh, off, size, ext, (n,))[0].get(n)
    rec['whole_member_sha256'] = whole_sha(fh, off, size)
    tf = one('TFLAG')
    rec['tflag_hours_ok'] = tflag_ok(tf, d) if tf is not None else False
    diffs = {}
    if kind == 'merged':
        for sp in SPECIES_M:
            a = one(sp)
            if a is None:
                diffs[sp] = 'absent'
                continue
            mine = means(a[:, 0]).astype(np.float32)
            del a
            diffs[sp] = ndiff(mine, z[sp])
            grid[0] = mine.shape
    else:
        g = groups[sec]
        r = g['row'] - BASE
        c = g['col'] - BASE
        nr, nc = grid[0]
        inside = (r >= 0) & (r < nr) & (c >= 0) & (c < nc)
        idx = (r * nc + c)[inside]
        pm = {}
        for sp in SPECIES_P:
            a = one(sp)
            if a is None:
                diffs[sp] = 'absent'
                continue
            pm[sp] = means(a[:, 0, :, 0])
            del a
            if pm[sp].size != r.size:
                diffs[sp] = 'stack count differs'
                continue
            cells = np.bincount(idx, weights=pm[sp][inside], minlength=nr * nc).reshape(nr, nc)
            diffs[sp] = ndiff(cells.astype(np.float32), z[sp])
        if 'P1_NOX' in z and all((isinstance(diffs.get(s), int) for s in SPECIES_P)):
            tot = pm['NO'] + pm['NO2'] + pm['HONO']
            cells = np.bincount(idx, weights=tot[inside], minlength=nr * nc).reshape(nr, nc)
            diffs['P1_NOX'] = ndiff(cells.astype(np.float32), z['P1_NOX'])
        rec['n_stacks_equal'] = int(z['n_stacks']) == int(r.size)
    fh.close()
    rec['elements_differing'] = diffs
    rec['seconds'] = round(time.time() - t0, 1)
    rec['ok'] = bool(rec['meta_equal'] and hok and rec['tflag_hours_ok'] and all((v == 0 for v in diffs.values())) and rec.get('n_stacks_equal', True))
    return rec

def read_groups(year, h_input):
    tre = '_%s_stackgroups_epicsoil_EQUATES_v1\\.0\\.tar$' % year
    out = {}
    for sec in DAILY + REP:
        tar, tpath, hits = find(year, h_input, tre, 'stack_groups_%s_12US1_' % sec)
        m = pick(hits)
        if m is None:
            continue
        name, size, off, ok, _ = m
        with open(tpath, 'rb') as fh:
            arrs, _ = read_vars(fh, off, size, name.rsplit('.', 1)[-1], ('ROW', 'COL'))
        out[sec] = {'row': np.asarray(arrs['ROW']).reshape(-1).astype(np.int64), 'col': np.asarray(arrs['COL']).reshape(-1).astype(np.int64), 'member': name}
    return out

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', required=True)
    ap.add_argument('--out-root', default='Research\\choi_emul\\extracts_c7a')
    ap.add_argument('--h-input', default='Research\\EQUATES_v1.0\\CMAQ_12US1\\INPUT')
    a = ap.parse_args(argv)
    y = a.year
    ydir = os.path.join(a.out_root, y)
    res = {'year': y, 'started': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'decoder': 'own tar walk; h5py chunk index only; zlib, own unshuffle; scipy for .ncf; own float64 means and bincount sums', 'members': []}
    try:
        grid = [None]
        groups = read_groups(y, a.h_input)
        sample = [('merged', None, y + md) for md in MERGED_MMDD]
        sample += [('pt', s, y + DAILY_MMDD) for s in DAILY]
        sample += [('pt', s, None) for s in REP]
        for kind, sec, d in sample:
            if kind == 'pt' and (grid[0] is None or sec not in groups):
                res['members'].append({'kind': kind, 'sector': sec, 'date': d, 'found': False, 'why': 'no grid or no groups'})
                continue
            res['members'].append(check_member(y, kind, sec, d, a.h_input, ydir, grid, groups))
        nd = 0
        for r in res['members']:
            for v in (r.get('elements_differing') or {}).values():
                nd += v if isinstance(v, int) and v > 0 else 0
        res['n_members'] = len(res['members'])
        res['n_ok'] = sum((1 for r in res['members'] if r.get('ok')))
        res['n_elements_differing'] = nd
        res['verdict'] = 'PASS' if res['n_ok'] == 10 and res['n_members'] == 10 else 'FAIL'
        rc = 0 if res['verdict'] == 'PASS' else 1
    except Exception as e:
        res['verdict'] = 'NOT RUN'
        res['error'] = '%s: %s' % (type(e).__name__, str(e)[:200])
        rc = 2
    res['ended'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
    os.makedirs(ydir, exist_ok=True)
    s = json.dumps(res, indent=1, sort_keys=True, ensure_ascii=True)
    with open(os.path.join(ydir, 'VALUECHECK_%s.json' % y), 'w', encoding='ascii', newline='\n') as f:
        f.write(s + '\n')
    lines = ['VALUE CHECK %s (separate decoder): %s; %s of %s members ok; %s elements differing' % (y, res['verdict'], res.get('n_ok'), res.get('n_members'), res.get('n_elements_differing'))]
    for r in res['members']:
        lines.append('%s %s %s: ok %s; differing %s; whole sha256 %s' % (r.get('kind'), r.get('sector') or '', r.get('date'), r.get('ok'), json.dumps(r.get('elements_differing')), (r.get('whole_member_sha256') or '')[:16]))
    out = []
    for ln in lines:
        out.extend(textwrap.wrap(ln, 78, subsequent_indent='  ', break_long_words=True))
    with open(os.path.join(ydir, 'VALUECHECK_%s_ASCII.txt' % y), 'w', encoding='ascii', newline='\n') as f:
        f.write('\n'.join(out) + '\n')
    print(out[0])
    return rc
if __name__ == '__main__':
    sys.exit(main())
