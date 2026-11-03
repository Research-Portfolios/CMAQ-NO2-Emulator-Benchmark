"""Reads members inside a local tar without unpacking it: a seekable window over one member that records the byte ranges read and their sha256, a member listing by the tar-header reader, and the archive checks."""
import hashlib
import io
import os
import c7a_common as C
tario, stacks, s3io = C.use_vendor()

class LocalWindow(io.RawIOBase):

    def __init__(self, fh, start, size):
        super().__init__()
        self.fh = fh
        self.start = int(start)
        self.size = int(size)
        self.pos = 0
        self.reads = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = int(off)
        elif whence == 1:
            self.pos += int(off)
        else:
            self.pos = self.size + int(off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), max(0, self.size - self.pos))
        if n <= 0:
            return 0
        self.fh.seek(self.start + self.pos)
        d = self.fh.read(n)
        b[:len(d)] = d
        if d:
            self.reads.append((self.pos, len(d)))
        self.pos += len(d)
        return len(d)

    def digest(self):
        iv = sorted(self.reads)
        merged = []
        for a, n in iv:
            e = a + n
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([a, e])
        h = hashlib.sha256()
        tot = 0
        for a, e in merged:
            self.fh.seek(self.start + a)
            left = e - a
            while left:
                d = self.fh.read(min(left, 8 << 20))
                if not d:
                    raise IOError('short re-read for the digest')
                h.update(d)
                left -= len(d)
            tot += e - a
        return (h.hexdigest(), tot, len(merged))

def read_member(fh, m):
    buf = bytearray(int(m['size']))
    mv = memoryview(buf)
    fh.seek(int(m['data_offset']))
    pos = 0
    h = hashlib.sha256()
    while pos < len(buf):
        n = fh.readinto(mv[pos:pos + (8 << 20)])
        if not n:
            raise IOError('short read of member %s' % m['name'])
        h.update(mv[pos:pos + n])
        pos += n
    mv.release()
    return (buf, h.hexdigest())

def walk(path):
    size = os.path.getsize(path)
    with open(path, 'rb') as fh:
        ms, w = tario.walk(tario.file_reader(fh), start=0, end=size, strict=True)
    return (ms, w)

def k2(path, pinned_bytes, ref_sizes):
    size = os.path.getsize(path)
    out = {'bytes': size, 'pinned_bytes': int(pinned_bytes), 'whole_blocks': size % 512 == 0, 'equal_to_pin': size == int(pinned_bytes), 'ref_2017_2019': list(ref_sizes)}
    flag = None
    if ref_sizes:
        lo, hi = (C.K2_LOW * min(ref_sizes), C.K2_HIGH * max(ref_sizes))
        flag = not lo <= size <= hi
    out['flag'] = flag
    out['ok'] = out['whole_blocks'] and out['equal_to_pin']
    return out

def k3(members, contents_bytes):
    entries, unparsed = s3io.parse_contents(contents_bytes.decode('utf-8', 'replace'))
    bad = []
    if len(entries) != len(members):
        bad.append('count: contents %d, headers %d' % (len(entries), len(members)))
    for i, (e, m) in enumerate(zip(entries, members)):
        why = []
        if s3io.norm_name(e['name']) != s3io.norm_name(m['name']):
            why.append('name')
        if not m['checksum_ok']:
            why.append('checksum')
        if e['type'] == '-':
            if not m['is_file'] or int(e['size']) != int(m['size']):
                why.append('size or type')
        elif e['type'] == 'd':
            if m['kind'] != 'dir':
                why.append('type')
        if why:
            bad.append('member %d %s: %s' % (i, s3io.norm_name(m['name']), ', '.join(why)))
    return {'ok': not bad and unparsed == 0, 'n_contents': len(entries), 'n_headers': len(members), 'unparsed': unparsed, 'mismatches': bad[:10], 'n_mismatches': len(bad)}

def pick(members, year):
    found, issues = ({}, [])
    for m in members:
        if not m['is_file']:
            continue
        c = C.classify(s3io.norm_name(m['name']), year)
        if c is None:
            continue
        kind, sector, date, ext = c
        if date is not None and date[:4] != str(year):
            issues.append('other-year member not used: %s' % m['name'])
            continue
        found.setdefault((kind, sector, date), []).append((ext, m))
    out = {}
    for key, lst in found.items():
        nc4 = [m for e, m in lst if e == 'nc4']
        use = nc4 or [m for e, m in lst]
        if len(use) > 1:
            issues.append('AMBIGUOUS %s: %d members' % (key, len(use)))
            continue
        out[key] = use[0]
        if len(lst) > 1:
            issues.append('%s: .nc4 taken over .ncf' % (key,))
    return (out, issues)
