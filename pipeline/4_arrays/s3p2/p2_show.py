"""Prints a JSON file's subtree with values shortened and at signs masked, or one key of each child (read only)."""
import json
import sys
sys.path.insert(0, '.\\run\\code')
from choi_dl import common

def main():
    a = sys.argv[1:]
    each = None
    if '--each' in a:
        i = a.index('--each')
        each = a[i + 1]
        a = a[:i] + a[i + 2:]
    with open(a[0], 'r', encoding='ascii') as f:
        p = json.load(f)
    key = a[1] if len(a) > 1 else ''
    n = int(a[2]) if len(a) > 2 else 300
    for k in [x for x in key.split('/') if x]:
        p = p[int(k)] if isinstance(p, list) else p[k]
    items = p.items() if isinstance(p, dict) else enumerate(p) if isinstance(p, list) else [('', p)]
    if '--len' in sys.argv:
        print('len %d' % len(p) if hasattr(p, '__len__') else 'no len')
        return
    for k, v in items:
        if each is not None:
            v = v.get(each) if isinstance(v, dict) else None
        s = common.mask(json.dumps(v, sort_keys=True))
        print('%s = %s' % (k, s if len(s) <= n else s[:n] + ' ...(%d)' % len(s)))
if __name__ == '__main__':
    main()
