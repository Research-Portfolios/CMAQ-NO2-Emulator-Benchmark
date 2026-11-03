"""Checks the version field of every provenance row against the expected EQUATES, CMAQ, MCIP and WRF versions and, for RSIG, the coverage name and the sha256 of its description; prints the counts, every differing row and the HTTP status counts (read only)."""
import json
import os
import re
import sys
CODE = '.\\run\\code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from choi_dl import common, dl_s3, prov, rsig
PLAN_R = os.path.join(common.RUN_CONFIG, 'S2A_PLAN_RSIG.json')

def main():
    common.set_temp()
    with open(PLAN_R, 'rb') as f:
        dsha = json.loads(f.read().decode('ascii'))['describe_sha256']
    bad = []
    n = 0
    for p, r in prov.load(rsig.RAW_RSIG).items():
        n += 1
        m = re.search('COVERAGE=([^&]+)', str(r.get('source', '')))
        cov = m.group(1) if m else None
        want = rsig.VERSION_FMT % (cov, dsha.get(cov)) if cov else None
        if r.get('version') != want:
            bad.append((p, r.get('version')))
    print('rsig rows %d' % n)
    checks = ((dl_s3.MCIP_DIR, lambda r: dl_s3.VERSION_MCIP), (dl_s3.S3META, lambda r: dl_s3.VERSION_MCIP if '/met/' in str(r.get('source')) else dl_s3.VERSION_EMIS), (dl_s3.SG_DIR, lambda r: dl_s3.VERSION_EMIS), (dl_s3.EM_DIR, lambda r: dl_s3.VERSION_EMIS), (dl_s3.PT_DIR, lambda r: dl_s3.VERSION_EMIS))
    for folder, fn in checks:
        rows = prov.load(folder)
        for p, r in rows.items():
            if r.get('version') != fn(r):
                bad.append((p, r.get('version')))
        print('%s rows %d' % (common.rel(folder), len(rows)))
    ne = prov.load(common.NATURAL_EARTH)
    for p, r in ne.items():
        if not str(r.get('version', '')).strip():
            bad.append((p, r.get('version')))
    print('%s rows %d' % (common.rel(common.NATURAL_EARTH), len(ne)))
    print('rows whose version differs: %d' % len(bad))
    for p, v in bad[:40]:
        print('  %s: %s' % (p, common.mask(v)[:100]))
    import collections
    for folder in (rsig.RAW_RSIG, dl_s3.MCIP_DIR, dl_s3.S3META, dl_s3.SG_DIR, dl_s3.EM_DIR, dl_s3.PT_DIR, common.NATURAL_EARTH):
        c = collections.Counter((str(r.get('http_status')) for r in prov.load(folder).values()))
        print('http_status %s: %s' % (common.rel(folder), json.dumps(dict(c), sort_keys=True)))
    return 0
if __name__ == '__main__':
    sys.exit(main())
