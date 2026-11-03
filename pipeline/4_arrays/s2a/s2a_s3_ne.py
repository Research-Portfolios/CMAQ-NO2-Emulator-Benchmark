"""Fetches the Natural Earth files, hashes each and writes its provenance row, checks by name the 48 contiguous states, the District of Columbia and the ten cities, and writes a summary JSON."""
import os
import sys
sys.path.insert(0, '/content/emul/run/code')
sys.path.insert(0, '/content/emul/run/code/s2a')
from choi_dl import common, ne, prov
import s2a_s3_common as C

def main():
    common.set_temp()
    t = C.timer()
    C.log('NE start; URL template %s' % ne.template())
    fetched = ne.fetch(log=C.log)
    files = ne.rows(fetched, log=C.log)
    chk = ne.checks()
    copies = prov.copy_to_run(common.NATURAL_EARTH, 'ne_natural_earth')
    tot = sum((b for _, b, _ in files))
    out = {'made': common.now_pair(), 'template': ne.template(), 'fetched': fetched, 'files': [{'path': common.rel(p), 'bytes': b, 'sha256': s} for p, b, s in files], 'n_files': len(files), 'bytes': tot, 'checks': chk, 'copies': [common.rel(c) for c in copies], 'seconds': t()}
    C.save_json(os.path.join(C.WORK_S3, 'ne.json'), out)
    st = chk['states']
    C.log('NE files %d, %d bytes; states found %d of %d, missing %s; other US records %d' % (len(files), tot, len(st['found']), len(ne.CONTIGUOUS_48_AND_DC), st['missing'], len(st['other_us_records'])))
    for c, v in chk['cities'].items():
        ch = v['chosen']
        C.log('NE city %s: %d record(s); chosen %s' % (c, v['n_records'], None if ch is None else '%s, %s, lat %.4f, lon %.4f, POP_MAX %s, POP_MIN %s' % (ch['name'], ch['adm1'], ch['lat'], ch['lon'], ch['pop_max'], ch['pop_min'])))
    C.log('NE done in %.1f s' % t())
if __name__ == '__main__':
    main()
