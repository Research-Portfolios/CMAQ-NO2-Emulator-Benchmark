"""Prints the details of the S3 download plan (read only): the MCIP files, the merged-emission days, the point and marine sectors, members and stack counts, the stack groups entries and the Natural Earth summary."""
import collections
import json
import os
RUN = '.\\run'

def mask(s):
    s = str(s).replace('@', '<at>')
    return s.replace('%40', '<pct40>')

def p(label, v):
    print('%s: %s' % (label, mask(json.dumps(v, sort_keys=True))[:1500]))

def main():
    path = os.path.join(RUN, 'config', 'S2A_PLAN_S3.json')
    with open(path, 'rb') as f:
        d = json.loads(f.read().decode('ascii'))
    print('== mcip members')
    for m in d['mcip']['members']:
        print('  ' + mask(json.dumps(m, sort_keys=True))[:400])
    em = d['emis_merged']
    dates = em['dates']
    print('== emis_merged: %d dates, first %s, last %s; route %s; day_set %s' % (len(dates), dates[0], dates[-1], em['route'], em['day_set']))
    by = collections.Counter((x[:6] for x in dates))
    print('  per month: ' + ', '.join(('%s %d' % kv for kv in sorted(by.items()))))
    k0 = dates[0]
    p('  member of %s' % k0, em['members'][k0])
    p('  species', em['species'])
    p('  units', em['units'])
    pm = d['pt_marine']
    print('== pt_marine')
    for k in ('route', 'sectors_kept', 'sectors_dropped', 'order', 'species', 'base', 'bases_found', 'both_tests_passed', 'members_per_month', 'holiday_members', 'stack_counts', 'groups_members', 'federal_holidays', 'n_members', 'bytes'):
        p('  ' + k, pm.get(k))
    ms = pm['members']
    c = collections.Counter(((m.get('sector'), m.get('year')) for m in ms))
    print('  members by sector and year: ' + ', '.join(('%s %s %d' % (a, b, n) for (a, b), n in sorted(c.items()))))
    p('  member[0]', ms[0])
    p('  member[-1]', ms[-1])
    keys = set()
    for m in ms:
        keys.update(m.keys())
    print('  member keys: %s' % sorted(keys))
    print('== stackgroups')
    for y, v in sorted(d['stackgroups'].items()):
        p('  ' + y, v)
    print('== natural_earth')
    ne = d['natural_earth']
    for k in ('n_files', 'bytes', 'states_found', 'states_missing', 'template'):
        p('  ' + k, ne.get(k))
if __name__ == '__main__':
    main()
