"""Writes and reads the provenance rows of each downloaded file or extract (path, source, byte range, retrieval time, HTTP status, size, sha256, version and related fields) as a text record and a JSON-lines record per folder, verifies a file against its row, and copies the records to the record folder."""
import hashlib
import json
import os
import shutil
from choi_dl import common
FIELDS = ('path', 'source', 'byte_range', 'retrieved_utc', 'retrieved_local', 'http_status', 'bytes', 'sha256', 'version', 'etag', 'last_modified', 'species', 'hours', 'member_sha256', 'member_note', 'note')

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def _check_text(v):
    s = str(v)
    s.encode('ascii')
    if '@' in s or '%40' in s.lower():
        raise ValueError('a provenance value holds an at sign or %40')
    return s

def append(folder, **row):
    unknown = set(row) - set(FIELDS)
    if unknown:
        raise ValueError('unknown provenance fields: %s' % sorted(unknown))
    if 'path' not in row:
        raise ValueError('a provenance row needs its path')
    row = {k: _check_text(v) for k, v in row.items() if v is not None}
    row['path'] = common.rel(row['path']) if os.path.isabs(row['path']) else row['path']
    os.makedirs(folder, exist_ok=True)
    jl = os.path.join(folder, 'PROVENANCE.jsonl')
    with open(jl, 'ab') as f:
        f.write((json.dumps(row, sort_keys=True) + '\n').encode('ascii'))
    lines = ['ROW ' + row['path'][-73:]]
    for k in FIELDS:
        if k in row:
            lines.extend(common.wrap('  %s: ' % k, row[k]))
    lines.append('')
    txt = os.path.join(folder, 'PROVENANCE_ASCII.txt')
    with open(txt, 'ab') as f:
        f.write(('\n'.join(lines) + '\n').encode('ascii'))
    return row

def load(folder):
    jl = os.path.join(folder, 'PROVENANCE.jsonl')
    rows = {}
    if os.path.exists(jl):
        with open(jl, 'rb') as f:
            for line in f.read().decode('ascii').split('\n'):
                if line.strip():
                    r = json.loads(line)
                    rows[r['path']] = r
    return rows

def verified(folder, path):
    r = load(folder).get(common.rel(path))
    if not r or 'sha256' not in r or 'bytes' not in r:
        return False
    if not os.path.exists(path) or os.stat(path).st_size != int(r['bytes']):
        return False
    return sha256_file(path) == r['sha256']

def copy_to_run(folder, name):
    os.makedirs(common.RUN_PROV, exist_ok=True)
    out = []
    for fn in ('PROVENANCE_ASCII.txt', 'PROVENANCE.jsonl'):
        src = os.path.join(folder, fn)
        if os.path.exists(src):
            dst = os.path.join(common.RUN_PROV, '%s_%s' % (name, fn))
            shutil.copyfile(src, dst)
            out.append(dst)
    return out
