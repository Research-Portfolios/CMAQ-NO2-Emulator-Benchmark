"""Prints the keys of the inventory and of the two download plans with short samples (read only)."""
import json
import os
import sys
CODE = '.\\run\\code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from choi_dl import common
CFG = os.path.join(common.RUN, 'config')

def short(v, n=300):
    s = json.dumps(v, sort_keys=True)
    s = common.mask(s)
    return s if len(s) <= n else s[:n] + ' ...(%d chars)' % len(s)

def show_inventory():
    with open(os.path.join(CFG, 'S2C_INVENTORY.json'), 'r', encoding='ascii') as f:
        inv = json.load(f)
    for k, v in inv.items():
        if k == 'files':
            print('files: %d entries' % len(v))
            continue
        if isinstance(v, dict):
            print('%s: dict keys %s' % (k, short(sorted(v.keys()), 400)))
            for k2, v2 in v.items():
                print('   %s.%s = %s' % (k, k2, short(v2, 400)))
        else:
            print('%s = %s' % (k, short(v, 600)))
    files = inv['files']
    if isinstance(files, dict):
        items = list(files.items())[:3]
        print('files sample:', short(items, 1200))
    else:
        kinds = {}
        for e in files:
            kinds.setdefault((e.get('kind'), e.get('role'), e.get('s3_input')), []).append(e)
        for key, lst in sorted(kinds.items(), key=lambda t: str(t[0])):
            print('KIND %s: %d; sample %s' % (key, len(lst), short(lst[0], 700)))

def show_plans():
    for name in ('S2A_PLAN_RSIG.json', 'S2A_PLAN_S3.json'):
        with open(os.path.join(CFG, name), 'r', encoding='ascii') as f:
            p = json.load(f)
        print('=== %s' % name)
        for k, v in p.items():
            print('%s = %s' % (k, short(v, 1500)))

def show_files(kind):
    with open(os.path.join(CFG, 'S2C_INVENTORY.json'), 'r', encoding='ascii') as f:
        inv = json.load(f)
    n = 0
    for e in inv['files']:
        if e.get('kind') == kind:
            print(short(e, 600))
            n += 1
            if n >= 12:
                break

def show_keys(name, path, n=400):
    with open(os.path.join(CFG, name), 'r', encoding='ascii') as f:
        p = json.load(f)
    for k in [x for x in path.split('.') if x]:
        p = p[k]
    if isinstance(p, dict):
        for k, v in p.items():
            print('%s = %s' % (k, short(v, n)))
    elif isinstance(p, list):
        print('list of %d; first %s' % (len(p), short(p[:3], n)))
    else:
        print(short(p, n))

def main():
    a = sys.argv[1:] or ['inventory']
    if a[0] == 'inventory':
        show_inventory()
    elif a[0] == 'plans':
        show_plans()
    elif a[0] == 'files':
        show_files(a[1])
    elif a[0] == 'keys':
        show_keys(a[1], a[2] if len(a) > 2 else '', int(a[3]) if len(a) > 3 else 400)
if __name__ == '__main__':
    main()
