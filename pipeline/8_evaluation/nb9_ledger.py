"""The spend ledger and guard of the evaluation runtime: sums the recorded compute units of every runtime record at the configured rates and stops when the total passes the limit."""
import glob
import json
import os
import time
AUTHORS_FORM = 'authors_form'

class Unreadable(Exception):
    pass

def _num(x):
    return isinstance(x, (int, float)) and (not isinstance(x, bool))

def guard_conf(conf):
    g = conf['guard']
    n = conf['nb9']['guard']
    return {'units': float(g['units']), 'missing_rate': float(g['missing_rate']), 'folders': list(n['folders']), 'units_file': n['units_file']}

def listed_rates(conf):
    out = [conf['cpu_units_per_hour']]
    out += [g['units_per_hour'] for g in conf['gpu_candidates']]
    return [float(x) for x in out if _num(x) and x > 0]

def _field(conf, acc):
    if acc == 'CPU':
        return conf['cpu_units_per_hour']
    for g in conf['gpu_candidates']:
        if g['key'] == acc:
            return g['units_per_hour']
    return AUTHORS_FORM

def runtime_name(conf, acc):
    if acc == 'CPU':
        return 'CPU'
    if acc in [g['key'] for g in conf['gpu_candidates']]:
        return acc
    return 'OTHER'

def rate_for(conf, rec):
    acc = (rec or {}).get('accelerator')
    f = _field(conf, acc)
    if _num(f) and f > 0:
        return (float(f), 'config')
    if f == AUTHORS_FORM:
        v = rec.get('units_per_hour_typed')
        if _num(v) and v > 0 and (rec.get('runtime_typed') == runtime_name(conf, acc)):
            return (float(v), 'typed')
    return (guard_conf(conf)['missing_rate'], 'missing')

def _cache_name(folder, name):
    return folder.replace('/', '__') + '__' + name

def _load(p):
    with open(p, 'rb') as f:
        d = json.loads(f.read().decode('ascii'))
    if not isinstance(d, dict):
        raise ValueError('not an object')
    h = d.get('hours')
    if h is None:
        h = 0.0
    if not _num(h) or h < 0:
        raise ValueError('hours is not a number of at least 0')
    return (d, float(h))

def read_ledger(emul, conf, cache_dir):
    gc = guard_conf(conf)
    os.makedirs(cache_dir, exist_ok=True)
    rows, never = ({}, [])
    for folder in gc['folders']:
        top = os.path.join(emul, *folder.split('/'))
        for p in sorted(glob.glob(os.path.join(top, '*.json'))):
            name = os.path.basename(p)
            sid = name[:-5]
            cp = os.path.join(cache_dir, _cache_name(folder, name))
            try:
                rec, hours = _load(p)
                state = 'read'
            except (OSError, ValueError, UnicodeDecodeError):
                try:
                    rec, hours = _load(cp)
                    state = 'unreadable_cached'
                except (OSError, ValueError, UnicodeDecodeError):
                    never.append('%s/%s' % (folder, name))
                    continue
            if state == 'read':
                tmp = cp + '.partial'
                with open(tmp, 'w', encoding='ascii', newline='\n') as f:
                    json.dump(rec, f)
                os.replace(tmp, cp)
            if state == 'read':
                rate, how = rate_for(conf, rec)
            else:
                rate, how = (gc['missing_rate'], 'unreadable')
            row = {'sid': sid, 'folder': folder, 'state': state, 'rec': rec, 'hours': hours, 'rate': rate, 'how': how, 'units': hours * rate}
            old = rows.get(sid)
            if old is None or (hours, row['units']) > (old['hours'], old['units']):
                rows[sid] = row
    return {'rows': sorted(rows.values(), key=lambda r: (str(r['rec'].get('vm_started_utc', '')), r['sid'])), 'never_read': never}

def spent(led):
    return sum((r['units'] for r in led['rows']))

def guard(emul, conf, cache_dir, this_rate, every_s, end_s=600.0):
    gc = guard_conf(conf)
    led = read_ledger(emul, conf, cache_dir)
    if led['never_read']:
        raise Unreadable(', '.join(led['never_read'][:4]))
    sp = spent(led)
    reserve = float(this_rate) * (3.0 * float(every_s) + float(end_s)) / 3600.0
    left = gc['units'] - sp
    marked = [r['sid'] for r in led['rows'] if r['how'] in ('missing', 'unreadable')]
    return {'hit': left <= reserve, 'spent': sp, 'left': left, 'reserve': reserve, 'units': gc['units'], 'marked': marked, 'n': len(led['rows'])}

def guard_text(g):
    return "THE GUARD: the benchmark's own spend reached the %g-unit runaway guard: %.2f units spent, %.2f left, reserve %.2f" % (g['units'], g['spent'], g['left'], g['reserve'])
HOW_TEXT = {'config': '', 'typed': ' (typed for it)', 'missing': ' (no rate: counted at %g)', 'unreadable': " (unreadable record: its last readable copy's hours, counted at %g)"}

def units_lines(conf, led, other_total, notebook):
    gc = guard_conf(conf)
    mr = gc['missing_rate']
    lines = ["UNITS OF THE BENCHMARK (the benchmark's own spend; the runaway guard of %g units). Rebuilt by %s's driver at every step from EMUL/%s/ and EMUL/%s/ (hours from each VM's /proc/uptime; each record at the config's rate for its runtime, else the rate typed for it, else %g, marked)." % (gc['units'], notebook, gc['folders'][0], gc['folders'][1], mr)]
    for r in led['rows']:
        how = HOW_TEXT[r['how']]
        if '%g' in how:
            how = how % mr
        s = r['rec']
        lines.append('%s session %s (%s) %s %s: %.3f h x %g = %.3f units%s; updated %s' % (s.get('vm_started_utc'), r['sid'], r['folder'], s.get('notebook'), s.get('accelerator'), r['hours'], r['rate'], r['units'], how, s.get('updated_utc')))
    for n in led['never_read']:
        lines.append('EMUL/%s: unreadable and never read in this session: not counted here; a failed guard period' % n)
    lines.append("the benchmark's own spend: %.3f units of the %g-unit guard (%d sessions%s); stages 1 to 5 (the accounting file): %s units, shown beside, not counted; made %s" % (spent(led), gc['units'], len(led['rows']), ', %d unreadable' % len(led['never_read']) if led['never_read'] else '', 'not stated' if other_total is None else '%.3f' % other_total, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    return lines
