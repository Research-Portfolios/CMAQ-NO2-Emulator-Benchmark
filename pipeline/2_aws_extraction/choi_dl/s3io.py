"""Client of the public CMAS bucket on Amazon Web Services: anonymous HEAD and ranged GET requests to one host, paced and retried, with refusals never retried; it parses tar contents files, reads members inside the archives, opens HDF5 members, and computes daily grid means and per-stack means."""
import asyncio
import bisect
import gc
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from urllib.parse import urlsplit
import numpy as np
CODE_DIR = '/content/emul/run/code'
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from choi import tario
from choi_dl import common
BUCKET = 'cmas-equates'
HOST = 'cmas-equates.s3.amazonaws.com'
REGION = 'us-east-1'
USER_AGENT_EXTRA = 'emul-research/1.0 (public EQUATES data; ranged reads at most 1 per second)'
RETRY_DELAYS = (30.0, 120.0, 600.0)
MIN_GAP_S = 1.0
MAX_IN_FLIGHT = 4
BLOCK = 1 << 20
FIRST_BLOCK = 4 << 20
MERGE_GAP = 3 << 20
STREAM_PIECE = 8 << 20
REFUSAL_STATUS = (401, 403, 405, 429)

class S3Error(Exception):
    kind = 'failed'

    def __init__(self, msg, status=None, code=None):
        super().__init__(common.mask(msg))
        self.status = status
        self.code = code

class Refused(S3Error):
    kind = 'refused'

class NotFound(S3Error):
    kind = 'notfound'

class Failed(S3Error):
    kind = 'failed'

class WrongHost(Exception):
    pass

class HeaderMismatch(Exception):
    pass

class Hang(Exception):
    pass

class Limiter:

    def __init__(self, min_gap=MIN_GAP_S, max_in_flight=MAX_IN_FLIGHT, not_before=0.0):
        self.min_gap = float(min_gap)
        self.max_in_flight = int(max_in_flight)
        self.last_start = float(not_before) - self.min_gap
        self.in_flight = 0
        self.cond = threading.Condition(threading.Lock())

    def acquire(self):
        with self.cond:
            while self.in_flight >= self.max_in_flight:
                self.cond.wait()
            self.in_flight += 1
            slot = max(time.time(), self.last_start + self.min_gap)
            self.last_start = slot
        d = slot - time.time()
        if d > 0:
            time.sleep(d)
        return slot

    def release(self):
        with self.cond:
            self.in_flight -= 1
            self.cond.notify()

    def take_over(self, last_start):
        with self.cond:
            self.last_start = max(self.last_start, float(last_start))

class Stats:

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.bytes = 0
        self.by_op = {}
        self.retries = 0

    def add(self, op, nbytes=0):
        with self.lock:
            self.requests += 1
            self.bytes += int(nbytes)
            self.by_op[op] = self.by_op.get(op, 0) + 1

    def snapshot(self):
        with self.lock:
            return {'requests': self.requests, 'bytes': self.bytes, 'by_op': dict(self.by_op), 'retries': self.retries}

def classify(exc):
    err = exc
    seen = 0
    while err is not None and seen < 5:
        resp = getattr(err, 'response', None)
        if isinstance(resp, dict) and 'ResponseMetadata' in resp:
            status = resp.get('ResponseMetadata', {}).get('HTTPStatusCode')
            code = (resp.get('Error') or {}).get('Code')
            text = '%s %s' % (status, code)
            if status in REFUSAL_STATUS:
                return ('refused', status, code, text)
            if status == 404 or code in ('NoSuchKey', 'NoSuchBucket'):
                return ('notfound', status, code, text)
            if status is not None and 500 <= int(status) <= 599:
                return ('retry', status, code, text)
            return ('failed', status, code, text)
        err = err.__cause__ or err.__context__
        seen += 1
    name = type(exc).__name__
    text = '%s: %s' % (name, str(exc)[:200])
    if isinstance(exc, WrongHost):
        return ('failed', None, 'WrongHost', text)
    retry_names = ('Timeout', 'Connection', 'ClientPayload', 'IncompleteRead', 'ServerDisconnected', 'EndpointConnection', 'ReadTimeout', 'ConnectTimeout', 'HTTPClientError', 'ResponseStreaming', 'ShortRead', 'ClientOSError', 'ResponseParser')
    chain = []
    e = exc
    while e is not None and len(chain) < 5:
        chain.append(type(e).__name__)
        e = e.__cause__ or e.__context__
    if any((any((r in n for r in retry_names)) for n in chain)) or isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return ('retry', None, name, text)
    if isinstance(exc, FileNotFoundError):
        return ('notfound', None, name, text)
    if isinstance(exc, PermissionError):
        return ('refused', None, name, text)
    return ('failed', None, name, text)

class ShortRead(Exception):
    pass

class S3Client:

    def __init__(self, limiter=None, log=None, endpoint_url=None):
        import s3fs
        from fsspec.asyn import sync
        self._sync = sync
        self.limiter = limiter or Limiter()
        self.stats = Stats()
        self.log = log or (lambda s: None)
        self.hosts = {}
        self.first_host = None
        self.first_url = None
        cfg = {'s3': {'addressing_style': 'virtual', 'us_east_1_regional_endpoint': 'legacy'}, 'retries': {'total_max_attempts': 1, 'mode': 'standard'}, 'connect_timeout': 30, 'read_timeout': 120, 'user_agent_extra': USER_AGENT_EXTRA, 'max_pool_connections': 8}
        self.endpoint_url = endpoint_url
        self.fs = s3fs.S3FileSystem(anon=True, endpoint_url=endpoint_url, client_kwargs={'region_name': REGION}, config_kwargs=cfg, cache_regions=False, use_listings_cache=False, max_concurrency=1, skip_instance_cache=True)
        self.fs.retries = 1
        self.fs.connect()
        self.fs.s3.meta.events.register('before-send.s3', self._before_send)

    def _before_send(self, request=None, **kwargs):
        url = getattr(request, 'url', '') or ''
        host = (urlsplit(url).hostname or '').lower()
        if self.first_host is None:
            self.first_host = host
            self.first_url = url.split('?')[0]
        self.hosts[host] = self.hosts.get(host, 0) + 1
        if host != HOST:
            raise WrongHost('request to host %s, not %s (stopped before it was sent)' % (host, HOST))
        return None

    def _attempts(self, op, fn, what):
        attempt = 0
        while True:
            self.limiter.acquire()
            err = None
            res = None
            try:
                res = fn()
            except Exception as e:
                err = e
            finally:
                self.limiter.release()
            if err is None:
                return res
            kind, status, code, text = classify(err)
            if kind == 'refused':
                raise Refused('%s %s refused: %s' % (op, what, text), status, code)
            if kind == 'notfound':
                raise NotFound('%s %s not found: %s' % (op, what, text), status, code)
            if kind != 'retry' or attempt >= len(RETRY_DELAYS):
                raise Failed('%s %s failed: %s' % (op, what, text), status, code)
            wait = RETRY_DELAYS[attempt]
            attempt += 1
            with self.stats.lock:
                self.stats.retries += 1
            self.log('RETRY %d of 3 for %s %s after %s; waiting %.0f s' % (attempt, op, what, text, wait))
            time.sleep(wait)

    def head(self, key):

        def fn():
            r = self._sync(self.fs.loop, self.fs._call_s3, 'head_object', Bucket=BUCKET, Key=key)
            self.stats.add('head', 0)
            return r
        r = self._attempts('HEAD', fn, key)
        lm = r.get('LastModified')
        return {'key': key, 'size': int(r.get('ContentLength')), 'etag': str(r.get('ETag', '')).strip('"'), 'last_modified': lm.strftime('%Y-%m-%dT%H:%M:%SZ') if hasattr(lm, 'strftime') else str(lm), 'http_status': int(r['ResponseMetadata']['HTTPStatusCode'])}

    async def _aget(self, key, start, end, sink, if_match, out):
        kw = {'Bucket': BUCKET, 'Key': key, 'Range': 'bytes=%d-%d' % (start, end - 1)}
        if if_match:
            kw['IfMatch'] = '"%s"' % if_match
        resp = await self.fs._call_s3('get_object', **kw)
        out['status'] = int(resp['ResponseMetadata']['HTTPStatusCode'])
        out['etag'] = str(resp.get('ETag', '')).strip('"')
        lm = resp.get('LastModified')
        out['last_modified'] = lm.strftime('%Y-%m-%dT%H:%M:%SZ') if hasattr(lm, 'strftime') else str(lm)
        out['content_range'] = resp.get('ContentRange')
        body = resp['Body']
        try:
            while True:
                b = await body.read(STREAM_PIECE)
                if not b:
                    break
                sink(b)
        finally:
            body.close()

    def get_range(self, key, start, end, sink, if_match=None, what=None):
        if end <= start:
            return {'status': None, 'bytes': 0}
        got = [0]

        def counting(b):
            got[0] += len(b)
            sink(b)
        info = {}
        attempts = [0]

        def fn():
            pos = start + got[0]
            before = got[0]
            out = {}
            attempts[0] += 1
            try:
                self._sync(self.fs.loop, self._aget, key, pos, end, counting, if_match, out)
            finally:
                self.stats.add('get', got[0] - before)
            info.update(out)
            if start + got[0] != end:
                raise ShortRead('ranged GET ended at %d of %d' % (start + got[0], end))
            return out
        self._attempts('GET', fn, what or '%s [%d, %d)' % (key, start, end))
        info['bytes'] = got[0]
        info['attempts'] = attempts[0]
        return info

    def get_bytes(self, key, start, end, if_match=None, what=None):
        buf = bytearray()
        info = self.get_range(key, start, end, buf.extend, if_match, what)
        return (bytes(buf), info)

    def get_small_object(self, key, size, if_match=None):
        return self.get_bytes(key, 0, int(size), if_match=if_match)

    def stream_to_file(self, key, start, end, path, if_match=None, what=None):
        part = path + '.partial'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        h = hashlib.sha256()
        t0 = time.time()
        with open(part, 'wb') as fh:

            def sink(b):
                h.update(b)
                fh.write(b)
            info = self.get_range(key, start, end, sink, if_match, what)
        if os.stat(part).st_size != end - start:
            raise Failed('stream to %s: %d bytes, expected %d' % (common.rel(part), os.stat(part).st_size, end - start))
        os.replace(part, path)
        info['sha256'] = h.hexdigest()
        info['seconds'] = time.time() - t0
        return info

    def stream_to_memory(self, key, start, end, if_match=None, what=None, copy=True):
        h = hashlib.sha256()
        buf = bytearray(end - start)
        mv = memoryview(buf)
        pos = [0]

        def sink(b):
            h.update(b)
            n = len(b)
            if pos[0] + n > len(buf):
                raise ShortRead('more bytes than the range asked for')
            mv[pos[0]:pos[0] + n] = b
            pos[0] += n
        t0 = time.time()
        try:
            info = self.get_range(key, start, end, sink, if_match, what)
        finally:
            mv.release()
        info['seconds'] = time.time() - t0
        info['sha256'] = h.hexdigest()
        if pos[0] != len(buf):
            raise Failed('stream to memory: %d of %d bytes' % (pos[0], len(buf)))
        return (bytes(buf) if copy else buf, info)

    def summary(self):
        s = self.stats.snapshot()
        s['hosts'] = dict(self.hosts)
        s['first_host'] = self.first_host
        s['first_url'] = self.first_url
        s['endpoint_url'] = self.endpoint_url
        return s
FALLBACK_ENDPOINT = 'https://s3.amazonaws.com'

def make_client(probe_key, log=None, limiter=None, endpoint_url=None):
    c = S3Client(limiter=limiter, log=log, endpoint_url=endpoint_url)
    try:
        h = c.head(probe_key)
        return (c, h, 'first request to %s (endpoint %s)' % (c.first_host, endpoint_url or 'botocore default'))
    except Failed as e:
        if e.code != 'WrongHost' or endpoint_url:
            raise
        first = c.first_host
        lim = c.limiter
        c2 = S3Client(limiter=lim, log=log, endpoint_url=FALLBACK_ENDPOINT)
        h = c2.head(probe_key)
        return (c2, h, 'the botocore default addressed %s (stopped before it was sent); endpoint set to %s, first request then to %s' % (first, FALLBACK_ENDPOINT, c2.first_host))
_CONTENTS_RE = re.compile('^(?P<mode>[-dlhcbps][-rwxsStTl]{9}[.+\\x40]?)\\s+(?P<og>\\S+)\\s+(?P<size>\\d+)\\s+(?P<date>\\d{4}-\\d{2}-\\d{2})\\s+(?P<time>\\d{2}:\\d{2}(?::\\d{2})?)\\s+(?P<name>.+?)\\s*$')

def parse_contents(text):
    out = []
    bad = 0
    for line in text.replace('\r\n', '\n').split('\n'):
        if not line.strip():
            continue
        m = _CONTENTS_RE.match(line)
        if not m:
            bad += 1
            continue
        mode = m.group('mode')
        out.append({'type': mode[0], 'size': int(m.group('size')), 'name': m.group('name')})
    return (out, bad)

def norm_name(n):
    n = n.strip()
    while n.startswith('./'):
        n = n[2:]
    return n.rstrip('/')

def expected_offsets(members):
    off = 0
    res = []
    for m in members:
        res.append(off)
        size = m['size'] if m['type'] in ('-',) else 0
        off += tario.BLOCK + tario.blocks(size)
    return res

class TarIndex:

    def __init__(self, client, key, etag, size, members):
        self.client = client
        self.key = key
        self.etag = etag
        self.size = int(size)
        self.members = members
        self.offsets = expected_offsets(members)
        self.by_name = {norm_name(m['name']): i for i, m in enumerate(members)}
        self.walked = None
        self.header_reads = 0

    def _read_range(self, off, n):
        if off >= self.size:
            return b''
        n = min(n, self.size - off)
        b, _ = self.client.get_bytes(self.key, off, off + n, if_match=self.etag, what='%s header at %d' % (self.key, off))
        self.header_reads += 1
        return b

    def locate(self, name):
        nn = norm_name(name)
        if nn not in self.by_name:
            raise HeaderMismatch('%s is not in the contents file' % nn)
        i = self.by_name[nn]
        want = self.members[i]
        off = self.offsets[i]
        w = tario.TarWalker(self._read_range, start=off, end=self.size, strict=False)
        first = None
        try:
            for m in w:
                first = m
                break
        except tario.TarHeaderError:
            first = None
        if first is not None and norm_name(first['name']) == nn and (first['size'] == want['size']) and first['checksum_ok'] and first['is_file']:
            return self._member(first, 'expected offset')
        if self.walked is None:
            ms, wk = tario.walk(self._read_range, start=0, end=self.size, strict=True)
            self.walked = {norm_name(m['name']): m for m in ms}
        m = self.walked.get(nn)
        if m is None or m['size'] != want['size'] or (not m['checksum_ok']):
            raise HeaderMismatch('%s: header (%s) does not match the contents file (size %d)' % (nn, 'missing' if m is None else 'size %d, checksum_ok %s' % (m['size'], m['checksum_ok']), want['size']))
        return self._member(m, 'header walk')

    @staticmethod
    def _member(m, how):
        return {'name': m['name'], 'size': int(m['size']), 'data_offset': int(m['data_offset']), 'header_offset': int(m['own_header_offset']), 'checksum_ok': bool(m['checksum_ok']), 'format': m['format'], 'how': how}

class MemberWindow(io.RawIOBase):

    def __init__(self, client, key, data_offset, size, etag=None, block=BLOCK, first_block=FIRST_BLOCK, gap=MERGE_GAP):
        super().__init__()
        self.client = client
        self.key = key
        self.base = int(data_offset)
        self.size = int(size)
        self.etag = etag
        self.block = int(block)
        self.first_block = int(first_block)
        self.gap = int(gap)
        self.pos = 0
        self.starts = []
        self.segs = []
        self.lock = threading.Lock()
        self.requests = 0
        self.nbytes = 0
        self.misses = 0
        self.prefetched = 0
        self.h = hashlib.sha256()
        self.order = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = int(off)
        elif whence == 1:
            self.pos += int(off)
        elif whence == 2:
            self.pos = self.size + int(off)
        else:
            raise ValueError('whence')
        return self.pos

    def _gaps(self, a, b):
        out = []
        cur = a
        with self.lock:
            i = bisect.bisect_right(self.starts, cur) - 1
            if i < 0:
                i = 0
            while cur < b and i < len(self.segs):
                s, e, _ = self.segs[i]
                if e <= cur:
                    i += 1
                    continue
                if s > cur:
                    out.append((cur, min(s, b)))
                cur = max(cur, e)
                i += 1
        if cur < b:
            out.append((cur, b))
        return [(x, y) for x, y in out if y > x]

    def _insert(self, s, data):
        with self.lock:
            i = bisect.bisect_left(self.starts, s)
            self.starts.insert(i, s)
            self.segs.insert(i, (s, s + len(data), data))

    def _fetch(self, a, b):
        data, info = self.client.get_bytes(self.key, self.base + a, self.base + b, if_match=self.etag, what='%s member bytes [%d, %d)' % (self.key, a, b))
        with self.lock:
            self.requests += 1
            self.nbytes += len(data)
            self.h.update(data)
            self.order.append((a, b))
        self._insert(a, data)

    def _assemble(self, a, b):
        parts = []
        cur = a
        with self.lock:
            i = bisect.bisect_right(self.starts, cur) - 1
            while cur < b:
                s, e, d = self.segs[i]
                if e <= cur:
                    i += 1
                    continue
                parts.append(d[cur - s:min(e, b) - s])
                cur = min(e, b)
                i += 1
        return b''.join(parts)

    def read_at(self, a, n):
        b = min(self.size, a + n)
        if a >= b:
            return b''
        gaps = self._gaps(a, b)
        if gaps:
            self.misses += 1
            lo = gaps[0][0] // self.block * self.block
            hi = -(-gaps[-1][1] // self.block) * self.block
            if lo == 0:
                hi = max(hi, self.first_block)
            hi = min(hi, self.size)
            for x, y in self._gaps(lo, hi):
                self._fetch(x, y)
        return self._assemble(a, b)

    def readinto(self, buf):
        n = len(buf)
        data = self.read_at(self.pos, n)
        k = len(data)
        buf[:k] = data
        self.pos += k
        return k

    def prefetch(self, ranges, workers=MAX_IN_FLIGHT):
        rs = sorted(((int(a), int(b)) for a, b in ranges if b > a))
        merged = []
        for a, b in rs:
            if merged and a - merged[-1][1] <= self.gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        todo = []
        for a, b in merged:
            todo.extend(self._gaps(a, min(b, self.size)))
        self.prefetched += len(todo)
        if not todo:
            return 0
        if workers <= 1 or len(todo) == 1:
            for a, b in todo:
                self._fetch(a, b)
            return len(todo)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as ex:
            futs = [ex.submit(self._fetch, a, b) for a, b in todo]
            for f in futs:
                f.result()
        return len(todo)

    def offset_order_digest(self):
        h = hashlib.sha256()
        with self.lock:
            segs = list(self.segs)
        cur = 0
        n = 0
        for s, e, d in segs:
            if e <= cur:
                continue
            a = max(s, cur)
            h.update(memoryview(d)[a - s:])
            n += e - a
            cur = e
        return (h.hexdigest(), n, len(segs))

    def report(self):
        dig, nb, nseg = self.offset_order_digest()
        return {'requests': self.requests, 'bytes': self.nbytes, 'misses': self.misses, 'prefetch_ranges': self.prefetched, 'sha256_of_bytes_read': self.h.hexdigest(), 'sha256_offset_order': dig, 'offset_order_bytes': nb, 'segments': nseg, 'member_size': self.size}

def attr_text(obj, name):
    v = obj.attrs.get(name) if hasattr(obj, 'attrs') else None
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode('utf-8', 'replace')
    elif isinstance(v, np.ndarray):
        if v.dtype.kind == 'S':
            v = b''.join(v.ravel().tolist()).decode('utf-8', 'replace')
        elif v.size == 1:
            v = v.ravel()[0]
            if isinstance(v, bytes):
                v = v.decode('utf-8', 'replace')
    return common.mask(str(v).strip())

class H5Var:

    def __init__(self, d):
        self._d = d
        self.shape = d.shape
        self.dimensions = ('TSTEP', 'LAY', 'ROW', 'COL') if len(d.shape) == 4 else tuple(('d%d' % i for i in range(len(d.shape))))

    def __getitem__(self, idx):
        return self._d[idx]

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        v = attr_text(self._d, name)
        if v is None:
            raise AttributeError(name)
        return v

class H5Vars:

    def __init__(self, f):
        self._f = f
        self._cache = {}

    def __contains__(self, name):
        return name in self._f

    def __getitem__(self, name):
        if name not in self._cache:
            self._cache[name] = H5Var(self._f[name])
        return self._cache[name]

class H5DS:

    def __init__(self, f):
        self.variables = H5Vars(f)

def needed_ranges(d, n_hours=24, layer=0):
    dsid = d.id
    shape = d.shape
    desc = {'shape': list(shape), 'chunks': list(d.chunks) if d.chunks else None, 'compression': d.compression, 'shuffle': bool(d.shuffle), 'dtype': str(d.dtype)}
    if d.chunks is None:
        off = dsid.get_offset()
        per = int(np.prod(shape[1:])) * d.dtype.itemsize
        desc['layout'] = 'contiguous'
        if off is None:
            return ([], desc)
        return ([(int(off), int(off) + n_hours * per)], desc)
    n = dsid.get_num_chunks()
    desc['layout'] = 'chunked'
    desc['n_chunks'] = int(n)
    out = []
    lc = d.chunks[1] if len(shape) > 1 else 1
    for i in range(n):
        ci = dsid.get_chunk_info(i)
        co = ci.chunk_offset
        if co[0] < n_hours and (len(shape) < 2 or co[1] <= layer < co[1] + lc):
            out.append((int(ci.byte_offset), int(ci.byte_offset) + int(ci.size)))
    desc['n_needed'] = len(out)
    desc['needed_bytes'] = int(sum((b - a for a, b in out)))
    return (out, desc)

def grid_means(f, species, n_hours=24, layer=0):
    from choi import stacks
    ds = H5DS(f)
    res = {'means': {}, 'units': {}, 'long_names': {}, 'var_desc': {}, 'missing': [], 'shapes': {}}
    res['hours'] = stacks.hours_check(ds, n_hours)
    for sp in species:
        if sp not in f:
            res['missing'].append(sp)
            continue
        d = f[sp]
        if len(d.shape) != 4 or d.shape[0] < n_hours:
            raise ValueError('%s has shape %s' % (sp, d.shape))
        a = d[0:n_hours, layer, :, :]
        res['means'][sp] = a.astype(np.float64).mean(axis=0).astype(np.float32)
        res['units'][sp] = attr_text(d, 'units')
        res['long_names'][sp] = attr_text(d, 'long_name')
        res['var_desc'][sp] = attr_text(d, 'var_desc')
        res['shapes'][sp] = list(d.shape)
    return res

def stack_means(f, species, n_hours=24):
    from choi import stacks
    ds = H5DS(f)
    hc = stacks.hours_check(ds, n_hours)
    dm = stacks.stack_daily_means(ds, [s for s in species if s in f], n_hours)
    long_names = {}
    for sp in species:
        if sp in f:
            long_names[sp] = attr_text(f[sp], 'long_name')
    return {'means': dm['means'], 'units': dm['units'], 'missing': [s for s in species if s not in f], 'n_stacks': dm['n_stacks'], 'n_nonfinite': dm['n_nonfinite'], 'n_negative': dm['n_negative'], 'hours': hc, 'long_names': long_names}

def file_attrs(f, names=('GDTYP', 'P_ALP', 'P_BET', 'P_GAM', 'XCENT', 'YCENT', 'XORIG', 'YORIG', 'XCELL', 'YCELL', 'NCOLS', 'NROWS', 'NLAYS', 'NVARS', 'SDATE', 'STIME', 'TSTEP')):
    out = {}
    for n in names:
        v = f.attrs.get(n)
        if v is None:
            continue
        a = np.asarray(v)
        if a.dtype.kind in 'iuf':
            out[n] = a.ravel().tolist()[0] if a.size == 1 else a.ravel().tolist()
    return out

def selective_read(client, key, member, etag, species, kind, n_hours=24, layer=0):
    import h5py
    t0 = time.time()
    win = MemberWindow(client, key, member['data_offset'], member['size'], etag)
    was = gc.isenabled()
    gc.disable()
    try:
        with h5py.File(win, 'r') as f:
            ranges = []
            layout = {}
            names = list(species) + ['TFLAG']
            for sp in names:
                if sp not in f:
                    continue
                rr, desc = needed_ranges(f[sp], n_hours if sp != 'TFLAG' else f[sp].shape[0], layer if sp != 'TFLAG' else 0)
                if sp == 'TFLAG':
                    rr = [(a, b) for a, b in rr]
                ranges.extend(rr)
                layout[sp] = desc
            meta_requests = win.requests
            meta_bytes = win.nbytes
            win.prefetch(ranges)
            pre_requests = win.requests
            if kind == 'grid':
                res = grid_means(f, species, n_hours, layer)
            else:
                res = stack_means(f, species, n_hours)
            res['attrs'] = file_attrs(f)
        res['window'] = win.report()
        res['window']['meta_requests'] = meta_requests
        res['window']['meta_bytes'] = meta_bytes
        res['window']['prefetch_requests'] = pre_requests - meta_requests
        res['window']['late_requests'] = win.requests - pre_requests
        res['layout'] = layout
        res['seconds'] = time.time() - t0
        return res
    finally:
        if was:
            gc.enable()

class MemReader(io.RawIOBase):

    def __init__(self, buf):
        super().__init__()
        self.mv = memoryview(buf)
        self.n = len(self.mv)
        self.pos = 0

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
        elif whence == 2:
            self.pos = self.n + int(off)
        return self.pos

    def readinto(self, b):
        k = max(0, min(len(b), self.n - self.pos))
        b[:k] = self.mv[self.pos:self.pos + k]
        self.pos += k
        return k

    def close(self):
        try:
            self.mv.release()
        except Exception:
            pass
        super().close()

def whole_read(client, key, member, etag, species, kind, n_hours=24, layer=0, data=None, zero_copy=False):
    import h5py
    t0 = time.time()
    if data is None:
        data, info = client.stream_to_memory(key, member['data_offset'], member['data_offset'] + member['size'], if_match=etag, what='%s member %s' % (key, member['name']), copy=not zero_copy)
    else:
        info = {'sha256': hashlib.sha256(data).hexdigest(), 'seconds': 0.0, 'bytes': len(data), 'attempts': 0}
    t1 = time.time()
    fobj = MemReader(data) if zero_copy else io.BytesIO(data)
    try:
        with h5py.File(fobj, 'r') as f:
            if kind == 'grid':
                res = grid_means(f, species, n_hours, layer)
            else:
                res = stack_means(f, species, n_hours)
            res['attrs'] = file_attrs(f)
    finally:
        fobj.close()
    res['transfer'] = {'bytes': len(data), 'sha256': info['sha256'], 'seconds': info['seconds'], 'attempts': info.get('attempts'), 'status': info.get('status')}
    res['seconds'] = time.time() - t0
    res['reduce_seconds'] = time.time() - t1
    return (res, data)

def _child_main(conn, code_dir, not_before):
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from choi_dl import common as cm
    cm.set_temp()
    client = None
    while True:
        try:
            task = conn.recv()
        except EOFError:
            break
        if task is None:
            break
        before = None
        try:
            if client is None:
                client = S3Client(limiter=Limiter(not_before=task.get('not_before', not_before)), endpoint_url=task.get('endpoint_url'))
            else:
                client.limiter.take_over(task.get('not_before', 0.0) - MIN_GAP_S)
            before = client.stats.snapshot()
            res = selective_read(client, task['key'], task['member'], task.get('etag'), task['species'], task['kind'], task.get('n_hours', 24), task.get('layer', 0))
            res['ok'] = True
        except Exception as e:
            kind, status, code, text = classify(e)
            res = {'ok': False, 'error_kind': getattr(e, 'kind', kind), 'error': cm.mask('%s: %s' % (type(e).__name__, str(e)[:400])), 'status': getattr(e, 'status', None) or status, 'code': cm.mask(str(getattr(e, 'code', None) or code or ''))}
        if client is not None:
            after = client.stats.snapshot()
            b0 = before or {'requests': 0, 'bytes': 0}
            res['requests_total'] = after['requests'] - b0['requests']
            res['bytes_total'] = after['bytes'] - b0['bytes']
            res['last_start'] = client.limiter.last_start
            res['hosts'] = dict(client.hosts)
            res['first_host'] = client.first_host
        try:
            conn.send(res)
        except Exception:
            break

class ReaderProcess:

    def __init__(self, parent_limiter, timeout=1800.0, endpoint_url=None):
        import multiprocessing as mp
        self.mp = mp.get_context('spawn')
        self.parent_limiter = parent_limiter
        self.endpoint_url = endpoint_url
        self.timeout = float(timeout)
        self.proc = None
        self.conn = None
        self.hangs = 0
        self.started = 0

    def _start(self):
        a, b = self.mp.Pipe()
        self.conn = a
        self.proc = self.mp.Process(target=_child_main, args=(b, CODE_DIR, self.parent_limiter.last_start + MIN_GAP_S), daemon=True)
        self.proc.start()
        self.started += 1

    def run(self, task, timeout=None):
        if self.proc is None or not self.proc.is_alive():
            self._start()
        task = dict(task)
        task['not_before'] = self.parent_limiter.last_start + MIN_GAP_S
        task['endpoint_url'] = self.endpoint_url
        self.conn.send(task)
        tmo = self.timeout if timeout is None else float(timeout)
        if self.conn.poll(tmo):
            res = self.conn.recv()
            if 'last_start' in res:
                self.parent_limiter.take_over(res['last_start'])
            return res
        self.hangs += 1
        try:
            self.proc.kill()
            self.proc.join(30)
        except Exception:
            pass
        self.proc = None
        self.parent_limiter.take_over(time.time())
        raise Hang('selective read of %s gave no answer in %.0f s; the child was killed' % (task['member']['name'], tmo))

    def close(self):
        if self.proc is not None:
            try:
                self.conn.send(None)
                self.proc.join(20)
            except Exception:
                pass
            if self.proc.is_alive():
                self.proc.kill()
            self.proc = None

def save_member(client, key, head, tindex, name, folder, version, note=None, dest_name=None, check=None):
    from choi_dl import prov
    nn = norm_name(name)
    path = os.path.join(folder, dest_name or nn.split('/')[-1])
    if prov.verified(folder, path):
        return (path, None, {'skipped': True})
    m = tindex.locate(name)
    info = client.stream_to_file(key, m['data_offset'], m['data_offset'] + m['size'], path, if_match=head['etag'], what='%s member %s' % (key, nn))
    if check is not None:
        try:
            info['check'] = check(path)
        except Exception as e:
            bad = path + '.failed_check'
            os.replace(path, bad)
            raise Failed('%s failed its check: %s: %s' % (nn, type(e).__name__, str(e)[:200]))
    u, loc = common.now_pair()
    prov.append(folder, path=path, source='s3://%s/%s (member %s)' % (BUCKET, key, nn), byte_range='%d-%d' % (m['data_offset'], m['data_offset'] + m['size'] - 1), retrieved_utc=u, retrieved_local=loc, http_status=info.get('status'), bytes=os.stat(path).st_size, sha256=info['sha256'], version=version, etag=head['etag'], last_modified=head['last_modified'], note=note)
    info['skipped'] = False
    return (path, m, info)

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return {'array': list(o.shape), 'dtype': str(o.dtype)}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, bytes):
        return '<%d bytes>' % len(o)
    return o

def dumps(o):
    return json.dumps(jsonable(o), sort_keys=True, indent=1)
