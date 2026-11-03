"""Reads tar headers by byte range without unpacking: member names (ustar, GNU long names and pax records), sizes, times, types and offsets, each header checksum verified, from any function that returns a byte range."""
BLOCK = 512
ZERO_BLOCK = bytes(BLOCK)
POSIX_MAGIC = b'ustar\x00'
GNU_MAGIC = b'ustar '
REGULAR_TYPES = (b'0', b'\x00', b'7', b'S')
SUPPORTED_TYPES = (b'0', b'\x00', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'L', b'K', b'S')
META_TYPES = (b'x', b'g', b'L', b'K')
PAX_KEYS_KEPT = ('path', 'size', 'mtime')
KINDS = {b'0': 'file', b'\x00': 'file', b'7': 'file', b'S': 'file', b'1': 'hardlink', b'2': 'symlink', b'3': 'chardev', b'4': 'blockdev', b'5': 'dir', b'6': 'fifo'}
MAX_META = 1 << 20

class TarHeaderError(ValueError):

    def __init__(self, offset, reason):
        super().__init__('tar header at byte %d: %s' % (offset, reason))
        self.offset = offset
        self.reason = reason

def blocks(n):
    return (int(n) + BLOCK - 1) // BLOCK * BLOCK

def _nts(b):
    i = b.find(b'\x00')
    return b if i < 0 else b[:i]

def _num(field, offset, what):
    if not field:
        raise TarHeaderError(offset, 'empty %s field' % what)
    if field[0] in (128, 255):
        n = int.from_bytes(field[1:], 'big')
        if field[0] == 255:
            n = -(256 ** (len(field) - 1) - n)
        return n
    s = _nts(field).strip(b' ')
    if not s:
        return 0
    try:
        return int(s.decode('ascii'), 8)
    except (UnicodeDecodeError, ValueError):
        raise TarHeaderError(offset, 'bad %s field' % what) from None

def checksums(block):
    rest = block[:148] + block[156:]
    unsigned = 256 + sum(rest)
    signed = 256 + sum((b - 256 if b > 127 else b for b in rest))
    return (unsigned, signed)

def _decode(b):
    return b.decode('utf-8', 'surrogateescape')

def parse_header(block, offset=0):
    if len(block) != BLOCK:
        raise TarHeaderError(offset, 'short header (%d bytes)' % len(block))
    if block == ZERO_BLOCK:
        return None
    stored = _num(block[148:156], offset, 'checksum')
    uns, sig = checksums(block)
    magic = block[257:263]
    if magic == POSIX_MAGIC:
        fmt = 'ustar'
    elif magic == GNU_MAGIC:
        fmt = 'gnu'
    else:
        fmt = 'v7'
    name = _nts(block[0:100])
    if fmt == 'ustar':
        prefix = _nts(block[345:500])
        if prefix:
            name = prefix + b'/' + name
    typeflag = block[156:157]
    h = {'name': _decode(name), 'size': _num(block[124:136], offset, 'size'), 'mtime': _num(block[136:148], offset, 'mtime'), 'typeflag': typeflag.decode('latin-1'), 'format': fmt, 'own_header_offset': int(offset), 'checksum_stored': stored, 'checksum_unsigned': uns, 'checksum_signed': sig, 'checksum_ok': stored in (uns, sig)}
    if h['size'] < 0:
        raise TarHeaderError(offset, 'negative size')
    return h

def parse_pax(data, offset=0):
    kept = {}
    pos = 0
    n = 0
    while pos < len(data) and data[pos] != 0:
        sp = data.find(b' ', pos)
        if sp < 0:
            raise TarHeaderError(offset, 'malformed pax record')
        try:
            length = int(data[pos:sp].decode('ascii'))
        except ValueError:
            raise TarHeaderError(offset, 'malformed pax length') from None
        if length <= sp - pos + 1 or pos + length > len(data):
            raise TarHeaderError(offset, 'pax record length out of range')
        rec = data[sp + 1:pos + length]
        if not rec.endswith(b'\n') or b'=' not in rec:
            raise TarHeaderError(offset, 'malformed pax record')
        key, _, val = rec[:-1].partition(b'=')
        k = key.decode('utf-8', 'replace')
        if k in PAX_KEYS_KEPT:
            kept[k] = val.decode('utf-8', 'surrogateescape')
        n += 1
        pos += length
    return (kept, n)

class TarWalker:

    def __init__(self, read_range, start=0, end=None, strict=True):
        self._read = read_range
        self.start = int(start)
        self.end = None if end is None else int(end)
        self.strict = strict
        self.end_offset = None
        self.zero_blocks = 0
        self.complete = False
        self.n_members = 0
        self.ranges_read = []
        self.bad_checksums = []

    def _get(self, off, n):
        if self.end is not None and off + n > self.end:
            n = max(0, self.end - off)
        self.ranges_read.append((int(off), int(n)))
        b = self._read(off, n) if n > 0 else b''
        return bytes(b)

    def __iter__(self):
        off = self.start
        global_pax = {}
        pending_pax = {}
        long_name = None
        chain = None
        while True:
            block = self._get(off, BLOCK)
            if len(block) < BLOCK:
                self.end_offset = off
                if self.strict:
                    raise TarHeaderError(off, 'end of object before the end-of-archive block')
                return
            h = parse_header(block, off)
            if h is None:
                self.end_offset = off
                self.zero_blocks = 1
                nxt = self._get(off + BLOCK, BLOCK)
                if nxt == ZERO_BLOCK:
                    self.zero_blocks = 2
                self.complete = True
                return
            if not h['checksum_ok']:
                self.bad_checksums.append(off)
                if self.strict:
                    raise TarHeaderError(off, 'header checksum mismatch: stored %d, unsigned %d, signed %d' % (h['checksum_stored'], h['checksum_unsigned'], h['checksum_signed']))
            t = h['typeflag'].encode('latin-1')
            if chain is None:
                chain = off
            if t in META_TYPES:
                if h['size'] > MAX_META:
                    raise TarHeaderError(off, 'meta header of %d bytes' % h['size'])
                data = self._get(off + BLOCK, h['size'])
                if len(data) < h['size']:
                    raise TarHeaderError(off, 'short meta header data')
                if t == b'x':
                    kept, _ = parse_pax(data, off)
                    pending_pax.update(kept)
                elif t == b'g':
                    kept, _ = parse_pax(data, off)
                    global_pax.update(kept)
                    chain = None
                elif t == b'L':
                    long_name = _decode(_nts(data))
                off += BLOCK + blocks(h['size'])
                continue
            eff = dict(global_pax)
            eff.update(pending_pax)
            name = h['name']
            if long_name is not None:
                name = long_name
            if 'path' in eff:
                name = eff['path']
            size = h['size']
            if 'size' in eff:
                size = int(eff['size'])
            mtime = h['mtime']
            if 'mtime' in eff:
                mtime = float(eff['mtime'])
            if t == b'\x00' and name.endswith('/'):
                t = b'5'
            if t == b'5':
                name = name.rstrip('/')
            data_offset = off + BLOCK
            m = {'name': name, 'size': size, 'mtime': mtime, 'typeflag': t.decode('latin-1'), 'kind': KINDS.get(t, 'other'), 'is_file': t in REGULAR_TYPES, 'format': h['format'], 'header_offset': chain, 'own_header_offset': off, 'data_offset': data_offset, 'checksum_ok': h['checksum_ok'], 'checksum_stored': h['checksum_stored'], 'checksum_unsigned': h['checksum_unsigned'], 'checksum_signed': h['checksum_signed'], 'long_name': long_name is not None, 'pax_keys': sorted(eff)}
            self.n_members += 1
            yield m
            if t in REGULAR_TYPES or t not in SUPPORTED_TYPES:
                off = data_offset + blocks(size)
            else:
                off = data_offset
            chain = None
            long_name = None
            pending_pax = {}

def walk(read_range, start=0, end=None, strict=True):
    w = TarWalker(read_range, start=start, end=end, strict=strict)
    return (list(w), w)

def bytes_reader(data):
    mv = memoryview(data)

    def read_range(offset, length):
        return bytes(mv[offset:offset + length])
    return read_range

def file_reader(fh):

    def read_range(offset, length):
        fh.seek(offset)
        return fh.read(length)
    return read_range
