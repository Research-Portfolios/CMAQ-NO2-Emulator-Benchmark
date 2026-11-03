"""The rules module of the residual trainings: the shared rules plus the GPU type taken as Colab assigns it, the two lanes, the checks of the design's training rows and step counts, and the guard that stops a run when its recorded compute-unit spend passes the limit."""
import datetime
import hashlib
import json
import re
DEC_ID = re.compile('^R-[0-9]{1,6}$')
SHA256 = re.compile('^[0-9a-f]{64}$')
NETWORK_KEYS = ('baseline', 'advanced', 'pixmlp')
GB = 1000000000.0
AUTHORS_FORM = 'authors_form'
UNITS_OTHER_LINE = 'the user states the units of stages 1 to 5 (EMUL/%s: absent or without a decided_by)'

def _rate_field_ok(v):
    return v is None or v == AUTHORS_FORM or (_num(v) and v > 0)

def load(path):
    with open(path, 'rb') as fh:
        raw = fh.read()
    return (json.loads(raw.decode('ascii')), hashlib.sha256(raw).hexdigest())

def is_dec(x):
    return isinstance(x, str) and bool(DEC_ID.match(x))

def _num(x):
    return isinstance(x, (int, float)) and (not isinstance(x, bool))

def form_errors(c):
    e = []
    if c.get('schema') != 'tier-a-config-1':
        e.append('schema is not tier-a-config-1')
    for k in ('design', 'decided_by', 'sealed', 'lockbox', 'c7a_applies', 'priorities', 'first_only', 'steps_main', 'steps_f6', 'width', 'day_sets', 'f19', 'hold_back_p1', 'reader', 'threads', 'python', 'torch', 'pins_train', 'pins_stage9', 'gpu_candidates', 'cpu_units_per_hour', 'choice_rule', 'unit_cap', 'units_other', 'copy_arrays', 'step_seconds_rule', 'departures', 'trainer_overrides', 'code_changes', 'stage9', 'places', 'first_only_then_stop'):
        if k not in c:
            e.append('%s is absent' % k)
        elif k not in c.get('sources', {}):
            e.append("%s has no source in 'sources'" % k)
    if e:
        return e
    if not isinstance(c['sealed'], bool):
        e.append('sealed is not true or false')
    pr = c['priorities']
    if not (isinstance(pr, list) and pr and all((isinstance(p, int) and 1 <= p <= 8 for p in pr))):
        e.append('priorities is not a list of 1 to 8')
    if not (isinstance(c['first_only'], list) and all((isinstance(s, str) for s in c['first_only']))):
        e.append('first_only is not a list of training names')
    for k in ('steps_main', 'steps_f6'):
        if k == 'steps_main' and isinstance(c[k], dict):
            e += steps_rule_errors(c[k])
            continue
        if not isinstance(c[k], int) or isinstance(c[k], bool):
            e.append('%s is not an integer' % k)
    if c.get('design') == P8_DESIGN:
        e += p8_form_errors(c)
    if c['width'] != 1.0:
        e.append('width is not 1 (w = 1 on the cloud)')
    if c['reader'] not in ('device', 'memmap'):
        e.append('reader is not device or memmap')
    for k in ('f19', 'hold_back_p1', 'first_only_then_stop'):
        if not isinstance(c[k], bool):
            e.append('%s is not true or false' % k)
    if c['threads'] is not None and (not (isinstance(c['threads'], int) and c['threads'] >= 1)):
        e.append('threads is not null or an integer of at least 1')
    t = c['torch']
    if not (isinstance(t, dict) and isinstance(t.get('version'), str) and isinstance(t.get('variants'), list) and t['variants'] and all((re.match('^cu[0-9]{3}$', str(v)) for v in t['variants']))):
        e.append('torch is not {version, variants [cuNNN, ...], index}')
    keys = []
    for g in c['gpu_candidates'] if isinstance(c['gpu_candidates'], list) else []:
        if not (isinstance(g, dict) and isinstance(g.get('key'), str) and isinstance(g.get('runtime_type'), str) and isinstance(g.get('match'), list) and g['match'] and isinstance(g.get('candidate'), bool) and _rate_field_ok(g.get('units_per_hour'))):
            e.append('a gpu_candidates entry is malformed')
            continue
        keys.append(g['key'])
    if not keys or len(set(keys)) != len(keys):
        e.append('gpu_candidates: no keys, or a key twice')
    cr = c['choice_rule']
    common = isinstance(cr, dict) and isinstance(cr.get('text'), str) and isinstance(cr.get('priorities'), list) and (cr.get('networks') == list(NETWORK_KEYS)) and _num(cr.get('memory_margin_gb')) and isinstance(cr.get('refuse_if_equivalence_fails'), bool) and isinstance(cr.get('never'), list)
    if common and cr.get('basis') == 'availability':
        cand = [g['key'] for g in c['gpu_candidates'] if isinstance(g, dict) and g.get('candidate') is True and (g.get('key') not in cr['never'])]
        od = cr.get('order')
        if not (isinstance(od, list) and od and (len(set(od)) == len(od)) and (sorted(od) == sorted(cand)) and isinstance(cr.get('one_type_per_training'), bool)):
            e.append('choice_rule is malformed (basis availability: order must name every candidate once; one_type_per_training true or false)')
    elif not (common and 'basis' not in cr and _num(cr.get('prefer_within_fraction')) and (cr.get('prefer') in keys)):
        e.append('choice_rule is malformed')
    if not (_num(c['unit_cap']) and c['unit_cap'] > 0):
        e.append('unit_cap is not a positive number')
    uo = c['units_other']
    if not (isinstance(uo, dict) and isinstance(uo.get('file'), str) and uo['file'].startswith('RUN/config/') and uo['file'].endswith('.json') and ('..' not in uo['file'])):
        e.append('units_other is not {file: RUN/config/<name>.json} (the accounting file of stages 1 to 5)')
    s9 = c['stage9']
    if not (isinstance(s9, dict) and 'decided_by' in s9 and (s9['decided_by'] is None or is_dec(s9['decided_by']))):
        e.append('stage9 has no decided_by (null or a id)')
    if c['copy_arrays'] not in ('after_choice', 'always'):
        e.append('copy_arrays is not after_choice or always')
    if c['step_seconds_rule'] not in ('slower_of_synthetic_and_real', 'synthetic'):
        e.append('step_seconds_rule is not slower_of_synthetic_and_real or synthetic')
    if not (isinstance(c['departures'], list) and all((isinstance(s, str) for s in c['departures']))):
        e.append('departures is not a list of strings')
    if not _rate_field_ok(c['cpu_units_per_hour']):
        e.append('cpu_units_per_hour is not null, a positive number or authors_form')
    return e

def refusals(c, stage='train'):
    out = ['the config: ' + x for x in form_errors(c)]
    if out:
        return out
    s9 = c.get('stage9') or {}
    if stage == 'train' or s9.get('needs_decided_by', True):
        if not is_dec(c['decided_by']):
            out.append('decided_by %r is not a decision id (R-<digits>): the design is not decided' % (c['decided_by'],))
    lb = c['lockbox']
    if not isinstance(lb, dict):
        out.append('lockbox is not {decided_by, file}')
    elif lb.get('decided_by') is None:
        if c['sealed'] is not True:
            out.append('sealed is not true while the lockbox is not decided ((c))')
    elif not is_dec(lb.get('decided_by')):
        out.append('lockbox.decided_by %r is not a decision id' % (lb.get('decided_by'),))
    elif lb.get('file') != 'LOCKBOX_DAYS.json':
        out.append("the lockbox is decided but lockbox.file is not LOCKBOX_DAYS.json (stage (8)'s d7 form)")
    elif not (SHA256.match(str(lb.get('sha256', ''))) and isinstance(lb.get('hashes'), dict) and all((SHA256.match(str(lb['hashes'].get(k, ''))) for k in ('lockbox', 'buffers')))):
        out.append('the lockbox is decided but lockbox.sha256 or lockbox.hashes {lockbox, buffers} is not a sha256')
    if stage == 'stage9' and c['sealed'] is not True:
        out.append("stage (9)'s metrics are sealed: sealed must be true")
    if stage == 'stage9' and (not is_dec(c['stage9'].get('decided_by'))):
        out.append("stage (9) is not decided: its baselines are refitted under the design's second step (stage9.decided_by null)")
    if c['first_only_then_stop'] and (not c['first_only']):
        out.append('first_only_then_stop is true but first_only is empty')
    ca = c['c7a_applies']
    if not (isinstance(ca, dict) and isinstance(ca.get('value'), bool)):
        out.append('c7a_applies.value is not a stated true or false (plan.py)')
    elif not is_dec(ca.get('decided_by')):
        out.append('c7a_applies.decided_by is not a decision id')
    if stage in ('train', 'stage9') and units_other_total(c) is None:
        out.append(UNITS_OTHER_LINE % c['units_other']['file'])
    for o in c['trainer_overrides'] if isinstance(c['trainer_overrides'], list) else [None]:
        if not (isinstance(o, dict) and isinstance(o.get('module'), str) and o['module'].startswith('train.') and isinstance(o.get('name'), str) and o['name'].isidentifier() and ('value' in o) and is_dec(o.get('decided_by'))):
            out.append('a trainer_overrides entry is not {module train.<m>, name, value, decided_by R-<digits>}')
    for o in c['code_changes'] if isinstance(c['code_changes'], list) else [None]:
        if not (isinstance(o, dict) and isinstance(o.get('src'), str) and isinstance(o.get('dst'), str) and o['dst'].startswith('code/') and ('..' not in o['dst']) and ('..' not in o['src']) and SHA256.match(str(o.get('sha256', ''))) and is_dec(o.get('decided_by'))):
            out.append('a code_changes entry is not {src, dst code/..., sha256, decided_by R-<digits>}')
    return out

def auto_departures(c):
    out = []
    if c['lockbox'].get('decided_by') is None:
        out.append('lockbox not decided at training time: no lockbox day is held out of any fit, validation, early stop or normalisation of this run')
    if c['sealed']:
        out.append('sealed: the held-out predictions go to PROJECT_DATA/EMUL/%s only, unread and unprinted until the study design is frozen by hash, then opened by its evaluation only' % c['places']['pred_sealed'])
    return out

def candidates(c):
    never = set(c['choice_rule']['never'])
    return [g for g in c['gpu_candidates'] if g['candidate'] and g['key'] not in never]

def rule_sha256(c):
    d = {'choice_rule': c['choice_rule'], 'candidates': sorted((g['key'] for g in candidates(c)))}
    b = json.dumps(d, sort_keys=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(b).hexdigest()

def device_mismatch(gpu, timing, tol_bytes=1000000000.0):
    g = gpu or {}
    t = timing or {}
    gm = int(g.get('total_memory_bytes') or 0)
    tmm = int(t.get('total_memory_bytes') or 0)
    if g.get('name') != t.get('device_name') or abs(gm - tmm) > tol_bytes:
        return {'session_name': g.get('name'), 'session_memory_bytes': gm, 'timed_name': t.get('device_name'), 'timed_memory_bytes': tmm}
    return None

def candidate_for(c, name):
    for g in c['gpu_candidates']:
        for m in g['match']:
            if re.search('(?<![A-Za-z0-9])%s(?![A-Za-z0-9])' % re.escape(m), str(name), re.I):
                return g
    return None

def typed_rate(session):
    v = (session or {}).get('units_per_hour_typed')
    return float(v) if _num(v) and v > 0 else None

def _rate_field(c, accelerator):
    if accelerator == 'CPU':
        return c['cpu_units_per_hour']
    for g in c['gpu_candidates']:
        if g['key'] == accelerator:
            return g['units_per_hour']
    return AUTHORS_FORM

def runtime_name(c, accelerator):
    if accelerator == 'CPU':
        return 'CPU'
    if accelerator in [g['key'] for g in c['gpu_candidates']]:
        return accelerator
    return 'OTHER'

def runtime_names(c):
    return [g['key'] for g in c['gpu_candidates']] + ['CPU', 'OTHER']

def typed_for_runtime(c, session):
    s = session or {}
    if s.get('runtime_typed') != runtime_name(c, s.get('accelerator')):
        return None
    return typed_rate(s)

def rate_of(c, accelerator, sessions=()):
    f = _rate_field(c, accelerator)
    if _num(f):
        return f
    if f != AUTHORS_FORM:
        return None
    best = None
    for s in sessions or ():
        if s.get('accelerator') != accelerator or typed_for_runtime(c, s) is None:
            continue
        k = (str(s.get('vm_started_utc', '')), str(s.get('updated_utc', '')))
        if best is None or k >= best[0]:
            best = (k, typed_for_runtime(c, s))
    return best[1] if best else None

def session_rate(c, session, sessions=()):
    s = session or {}
    if _rate_field(c, s.get('accelerator')) == AUTHORS_FORM:
        r = typed_for_runtime(c, s)
        if r is not None:
            return r
    return rate_of(c, s.get('accelerator'), sessions)

def rate_refusal(c, session):
    s = session or {}
    acc = s.get('accelerator')
    if _rate_field(c, acc) != AUTHORS_FORM:
        return None
    want = runtime_name(c, acc)
    if typed_rate(s) is None:
        return "no units per hour typed for this runtime (%s): type the compute units per hour that Colab shows for it in cell 1's field UNITS_PER_HOUR and choose %s in its field RUNTIME_SHOWN (Colab shows the rate in its Resources panel: click the RAM and Disk meters at the top right), then choose Run all" % (acc, want)
    if s.get('runtime_typed') != want:
        return "the rate in cell 1 was typed for %s, this runtime is %s: type this runtime's rate in UNITS_PER_HOUR and choose %s in RUNTIME_SHOWN, then choose Run all" % (s.get('runtime_typed') or 'no runtime', acc, want)
    return None

def units_other_ok(doc):
    return isinstance(doc, dict) and is_dec(doc.get('decided_by')) and isinstance(doc.get('sessions'), list) and _num(doc.get('total_units')) and (doc['total_units'] >= 0)

def attach_units_other(c, doc):
    out = dict(c)
    out['units_other_doc'] = doc
    return out

def units_other_total(c):
    doc = c.get('units_other_doc')
    return float(doc['total_units']) if units_other_ok(doc) else None

def projected_hours(c, step_seconds):
    from train import plan as P
    pl = {'step_seconds': dict(step_seconds), 'steps_main': steps_value(c), 'steps_f6': c['steps_f6']}
    tot, missing = (0.0, [])
    for t in P.trainings_for(c['choice_rule']['priorities'], c['c7a_applies']['value']):
        s = P.projected_seconds(t, pl)
        if s is None:
            missing.append(t['name'])
        else:
            tot += s
    return (tot / 3600.0, missing)

def choose(c, timings, equivs, sessions=()):
    rule = c['choice_rule']
    cands = [dict(g, units_per_hour=rate_of(c, g['key'], sessions)) for g in candidates(c)]
    failed = set((g['key'] for g in cands if rule['refuse_if_equivalence_fails'] and equivs.get(g['key']) is not None and (not equivs[g['key']].get('pass'))))
    for g in cands:
        if g['key'] in failed:
            continue
        if timings.get(g['key']) is None or equivs.get(g['key']) is None:
            return {'status': 'untimed', 'next': g}
    rh = rule_sha256(c)
    changed = sorted((g['key'] for g in cands if g['key'] not in failed and timings[g['key']].get('rule_sha256') != rh))
    if changed:
        return {'status': 'rule_changed', 'keys': changed}
    missing = sorted((g['key'] for g in cands if g['key'] not in failed and (not _num(g['units_per_hour']))))
    if missing:
        return {'status': 'rates_missing', 'keys': missing}
    rows = {}
    for g in cands:
        k = g['key']
        if k in failed:
            rows[k] = {'projected_hours': None, 'projected_units': None, 'units_per_hour': g['units_per_hour'], 'peak_gb': None, 'memory_gb': None, 'refused': ['the equivalence test did not pass']}
            continue
        t = timings[k]
        why = []
        peak = int(t['peak_allocated_max_bytes'])
        total = int(t['total_memory_bytes'])
        if peak > total - rule['memory_margin_gb'] * GB:
            why.append('peak %.2f GB with the device inputs above %.2f GB less %.1f GB' % (peak / GB, total / GB, rule['memory_margin_gb']))
        hours, miss = projected_hours(c, t['step_seconds'])
        if miss:
            why.append('no step time for %d trainings' % len(miss))
        rows[k] = {'projected_hours': round(hours, 3), 'units_per_hour': g['units_per_hour'], 'projected_units': round(hours * g['units_per_hour'], 2), 'peak_gb': round(peak / GB, 3), 'memory_gb': round(total / GB, 3), 'refused': why}
    ok = [k for k in rows if not rows[k]['refused']]
    if not ok:
        return {'status': 'none_eligible', 'rows': rows, 'rule_sha256': rh}
    best = min(ok, key=lambda k: (rows[k]['projected_units'], k))
    chosen = best
    pref = rule.get('prefer')
    if pref in ok and rows[pref]['projected_units'] <= rows[best]['projected_units'] * (1.0 + rule['prefer_within_fraction']):
        chosen = pref
    return {'status': 'chosen', 'chosen': chosen, 'fewest': best, 'rows': rows, 'rule_sha256': rh}

def spent_units(c, sessions):
    other = units_other_total(c)
    total = other or 0.0
    missing = [] if other is not None else ['stages 1 to 5']
    for s in sessions:
        r = session_rate(c, s, sessions)
        if not _num(r):
            missing.append(str(s.get('accelerator')))
            continue
        total += float(s.get('hours') or 0.0) * float(r)
    return (total, sorted(set(missing)))

def budget(c, sessions, gpu_key, prior_h):
    spent, missing = spent_units(c, sessions)
    if missing:
        raise ValueError('units not known for %s' % ', '.join(missing))
    rate = rate_of(c, gpu_key, sessions)
    if not _num(rate):
        raise ValueError('no units per hour for %s' % gpu_key)
    left = float(c['unit_cap']) - spent
    if left <= 0:
        raise ValueError('the cap of %g units is reached (%.2f spent)' % (c['unit_cap'], spent))
    return {'spent_units': round(spent, 3), 'left_units': round(left, 3), 'rate': rate, 'prior_hours': round(prior_h, 4), 'budget_hours': round(left / float(rate) + prior_h, 4)}

def cap_guard(c, sessions, gpu_key, every_s, end_s=600.0):
    if isinstance(c.get('guard'), dict):
        return guard_check(c, sessions, gpu_key, every_s, end_s)
    spent, missing = spent_units(c, sessions)
    rate = rate_of(c, gpu_key, sessions)
    reserve = None
    if _num(rate):
        reserve = float(rate) * (3.0 * float(every_s) + float(end_s)) / 3600.0
    else:
        missing = sorted(set(missing) | {str(gpu_key)})
    left = float(c['unit_cap']) - spent
    hit = bool(missing) or reserve is None or left <= reserve
    return {'hit': hit, 'spent': spent, 'left': left, 'reserve': reserve, 'rate': rate, 'missing': missing}

def stage9_cap(c, sessions):
    spent, miss = spent_units(c, sessions)
    if miss or spent >= float(c['unit_cap']):
        return 'the cap: %.2f of %g units spent%s; nothing runs' % (spent, c['unit_cap'], ' (units missing: %s)' % ', '.join(miss) if miss else '')
    return None

def lockbox_days(doc):
    days = doc.get('days', doc) if isinstance(doc, dict) else None
    if not isinstance(days, dict):
        raise ValueError('LOCKBOX_DAYS.json is not year -> day indices')
    out = {}
    for k, v in days.items():
        if not (isinstance(k, str) and len(k) == 4 and k.isdigit()):
            continue
        if not isinstance(v, list) or not all((isinstance(i, int) and (not isinstance(i, bool)) for i in v)):
            raise ValueError('LOCKBOX_DAYS.json %s: not a list of day indices' % k)
        out[int(k)] = set(v)
    if not out:
        raise ValueError('LOCKBOX_DAYS.json names no year')
    return out

def lockbox_only(doc):
    if not (isinstance(doc, dict) and isinstance(doc.get('lockbox'), dict)):
        raise ValueError('LOCKBOX_DAYS.json has no "lockbox" map')
    out = lockbox_days({'days': doc['lockbox']})
    days = lockbox_days(doc)
    for y, s in out.items():
        if not s <= days.get(y, set()):
            raise ValueError('LOCKBOX_DAYS.json %d: lockbox days outside "days"' % y)
    return out

def iso_date(s):
    t = str(s)
    if re.match('^[0-9]{8}$', t):
        t = '%s-%s-%s' % (t[:4], t[4:6], t[6:])
    if not re.match('^[0-9]{4}-[0-9]{2}-[0-9]{2}$', t):
        raise ValueError('date %r is not YYYYMMDD or YYYY-MM-DD' % (s,))
    datetime.date(int(t[:4]), int(t[5:7]), int(t[8:]))
    return t

def lockbox_dates(doc, arrays):
    days, lock = (lockbox_days(doc), lockbox_only(doc))
    years = arrays.get('years') or {}
    lk, bf = ([], [])
    for y in sorted(days):
        ds = (years.get(str(y)) or {}).get('dates')
        if not isinstance(ds, list):
            raise ValueError('S3_ARRAYS.json has no dates for %d' % y)
        for i in sorted(days[y]):
            if not 0 <= i < len(ds):
                raise ValueError("index %d of %d is outside S3_ARRAYS.json's %d dates" % (i, y, len(ds)))
            d = iso_date(ds[i])
            if d[:4] != str(y):
                raise ValueError('index %d of %d is the date %s' % (i, y, d))
            (lk if i in lock.get(y, ()) else bf).append(d)
    return (sorted(lk), sorted(bf))

def dates_sha256(dates):
    b = ''.join((d + '\n' for d in sorted(dates))).encode('ascii')
    return hashlib.sha256(b).hexdigest()

def lb1_errors(c, raw, arrays):
    lb = c['lockbox']
    out = []
    sha = hashlib.sha256(raw).hexdigest()
    if sha != lb.get('sha256'):
        out.append("LOCKBOX_DAYS.json sha256 %s, the config's %s" % (sha[:16], str(lb.get('sha256'))[:16]))
    try:
        lk, bf = lockbox_dates(json.loads(raw.decode('ascii')), arrays)
    except (ValueError, UnicodeDecodeError) as e:
        return out + ["the lockbox's dates: %s" % e]
    for name, ds in (('lockbox', lk), ('buffers', bf)):
        h = dates_sha256(ds)
        want = str((lb.get('hashes') or {}).get(name))
        if h != want:
            out.append("the %s dates (%d) hash to %s, the config's %s" % (name, len(ds), h[:16], want[:16]))
    return out

def lockbox_conflicts(days_by_training, lb):
    out = {}
    for name, days in days_by_training.items():
        n = sum((1 for y, i in days if int(i) in lb.get(int(y), ())))
        if n:
            out[name] = n
    return out
P8_DESIGN = 'STUDY_DESIGN'
STEPS_SM = 16000
STEPS_OTHER = 6000
SM_RULE = "S_M 16,000, G-7's branch when the budget does not bind"
GUARD_UNITS = 700.0
GUARD_RULED_BY = ['R-03', 'R-04']
LANES = ('1', '2')
LANES_RULED_BY = ''
PD25_DROP_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
GUARD_LINE = "THE GUARD: the benchmark's own spend reached the 700-unit runaway guard"

class GuardUnreadable(Exception):
    pass

def steps_rule_errors(r):
    e = []
    if r.get('rule') != SM_RULE:
        e.append('steps_main.rule is not the decided text')
    if r.get('steps_sm') != STEPS_SM:
        e.append('steps_main.steps_sm is not 16000')
    if r.get('other_steps') != STEPS_OTHER:
        e.append('steps_main.other_steps is not 6000')
    extra = sorted(set(r) - {'rule', 'steps_sm', 'other_steps'})
    if extra:
        e.append('steps_main has keys of R0 (removed): %s' % extra)
    return e

def steps_value(c):
    s = c['steps_main']
    return int(s['other_steps']) if isinstance(s, dict) else int(s)

def steps_sm_value(c):
    return int(c['steps_main']['steps_sm'])

def max_rate(c):
    vals = [g.get('units_per_hour') for g in c.get('gpu_candidates') or [] if isinstance(g, dict)] + [c.get('cpu_units_per_hour')]
    nums = [float(v) for v in vals if _num(v) and v > 0]
    return max(nums) if nums else None

def p8_form_errors(c):
    e = []
    if not isinstance(c['steps_main'], dict):
        e.append('steps_main is not the decided form')
    if c.get('first_only') != [] or c.get('first_only_then_stop') is not False:
        e.append("first_only is not empty or first_only_then_stop is not false (the study design's second step)")
    if c.get('c_channel') not in (True, 'offset', 'direct'):
        e.append('c_channel is not true, offset or direct')
    do = c.get('drop_order')
    if not (isinstance(do, list) and [x.get('id') for x in do if isinstance(x, dict)] == list(PD25_DROP_IDS) and all((isinstance(x.get('names'), list) and x['names'] and all((isinstance(n, str) for n in x['names'])) for x in do))):
        e.append('drop_order is not the items (1) to (8) with their training names')
    nd = c.get('never_dropped')
    if not (isinstance(nd, list) and nd and all((isinstance(n, str) for n in nd))):
        e.append('never_dropped is not a list of training names')
    if 'tail_rule' in c:
        e.append("tail_rule is the study design's run rule, which no longer decides")
    g = c.get('guard')
    mr = max_rate(c)
    if not (isinstance(g, dict) and _num(g.get('units')) and (float(g['units']) == GUARD_UNITS) and _num(c.get('unit_cap')) and (float(c['unit_cap']) == GUARD_UNITS) and _num(g.get('missing_rate')) and (mr is not None) and (float(g['missing_rate']) == mr) and isinstance(g.get('counts'), str) and (g.get('decided_by') == GUARD_RULED_BY)):
        e.append("guard is not {units 700 (= unit_cap), counts, missing_rate = the config's highest rate, decided_by [, ]}")
    ln = c.get('lanes')
    if not (isinstance(ln, dict) and all((isinstance(ln.get(k), list) and ln[k] and all((isinstance(n, str) for n in ln[k])) for k in LANES)) and isinstance(ln.get('rule'), str) and (ln.get('decided_by') == LANES_RULED_BY) and isinstance(ln.get('early_8'), dict) and all((isinstance(ln['early_8'].get(k), list) for k in LANES))):
        e.append('lanes is not {1: [names], 2: [names], rule, decided_by, early_8 {1, 2}}')
    for k in ('c_channel', 'drop_order', 'never_dropped', 'design_file', 'design_sha256', 'guard', 'lanes'):
        if k not in c.get('sources', {}):
            e.append("%s has no source in 'sources'" % k)
    return e

def p8_data_errors(c, drop_order, never_dropped):
    e = []
    want = [{'id': d, 'names': list(n)} for d, _t, n in drop_order]
    got = [{'id': x.get('id'), 'names': list(x.get('names') or [])} for x in c.get('drop_order') or []]
    if got != want:
        e.append("the config's drop_order differs from train.plan's")
    if sorted(c.get('never_dropped') or []) != sorted(never_dropped):
        e.append("the config's never_dropped differs from train.plan's")
    return e

def lanes_errors(c, lanes):
    ln = c.get('lanes') or {}
    e = []
    for k in LANES:
        if list(ln.get(k) or []) != list(lanes.get(k) or []):
            e.append("lane %s differs from the lanes' rule (train.plan.design_lanes)" % k)
        if list((ln.get('early_8') or {}).get(k) or []) != list((lanes.get('early_8') or {}).get(k) or []):
            e.append("lane %s's share of the first training stage's 8 differs from the rule's" % k)
    a, b = (set(ln.get('1') or []), set(ln.get('2') or []))
    if a & b:
        e.append('the lanes share trainings: %s' % sorted(a & b)[:4])
    return e

def lane_names(c, lane):
    return list(c['lanes'][str(lane)])

def lane_early(c, lane):
    return list(c['lanes']['early_8'][str(lane)])

def missing_rate(c):
    v = (c.get('guard') or {}).get('missing_rate')
    return float(v) if _num(v) and v > 0 else max_rate(c)

def guard_session_rate(c, s, sessions=()):
    s = s or {}
    mr = missing_rate(c)
    if s.get('never_read'):
        return (None, 'unreadable and never read by this session: a failed guard period')
    if s.get('unreadable'):
        return (mr, "unreadable now: its last readable copy's hours counted at %g" % mr)
    f = _rate_field(c, s.get('accelerator'))
    r = f if _num(f) else typed_for_runtime(c, s) if f == AUTHORS_FORM else None
    if _num(r) and r > 0:
        return (float(r), '')
    return (mr, '(no rate: counted at %g)' % mr)

def guard_spent(c, sessions):
    total, rows, never = (0.0, [], [])
    for s in sessions:
        r, note = guard_session_rate(c, s, sessions)
        rows.append((s, r, note))
        if r is None:
            never.append(str(s.get('session')))
            continue
        total += float(s.get('hours') or 0.0) * float(r)
    return (total, rows, never)

def guard_check(c, sessions, gpu_key, every_s, end_s=600.0):
    spent, rows, never = guard_spent(c, sessions)
    if never:
        raise GuardUnreadable('session records unreadable and never read: %s' % ', '.join(never[:6]))
    rate = rate_of(c, gpu_key, sessions)
    if not (_num(rate) and rate > 0):
        rate = missing_rate(c)
    reserve = float(rate) * (3.0 * float(every_s) + float(end_s)) / 3600.0
    units = float(c['guard']['units'])
    left = units - spent
    high = sorted(set((str(s.get('accelerator')) for s, r, note in rows if note)))
    return {'hit': left <= reserve, 'spent': spent, 'left': left, 'reserve': reserve, 'rate': float(rate), 'missing': [], 'units': units, 'counted_high': high, 'other_shown': units_other_total(c)}

def availability(c):
    return (c.get('choice_rule') or {}).get('basis') == 'availability'

def order_of(c):
    cs = candidates(c)
    if not availability(c):
        return cs
    by = dict(((g['key'], g) for g in cs))
    return [by[k] for k in c['choice_rule']['order'] if k in by]

def _peak_refusal(c, timing):
    rule = c['choice_rule']
    peak = int(timing['peak_allocated_max_bytes'])
    total = int(timing['total_memory_bytes'])
    if peak > total - rule['memory_margin_gb'] * GB:
        return 'peak %.2f GB with the device inputs above %.2f GB less %.1f GB' % (peak / GB, total / GB, rule['memory_margin_gb'])
    return None

def type_refusal(c, timing, equiv):
    rule = c['choice_rule']
    if rule['refuse_if_equivalence_fails'] and equiv is not None and (not equiv.get('pass')):
        return 'the equivalence test did not pass'
    if timing is not None and timing.get('rule_sha256') == rule_sha256(c):
        return _peak_refusal(c, timing)
    return None

def change_line(c, refused=()):
    left = [g for g in order_of(c) if g['key'] not in set(refused)]
    if not left:
        return 'no candidate GPU is left (%s refused by their checks)' % ', '.join((g['key'] for g in order_of(c)))
    first = left[0]
    if len(left) == 1:
        return 'Runtime > Change runtime type > %s with High-RAM on, keep the LANE in cell 1, then Run all' % first['runtime_type']
    return 'Runtime > Change runtime type > %s with High-RAM on; if Colab does not give it, %s with High-RAM on; keep the LANE in cell 1, then Run all' % (first['runtime_type'], ' or else '.join((g['runtime_type'] for g in left[1:])))

def type_split(c, key, states, names):
    one = availability(c) and bool(c['choice_rule'].get('one_type_per_training'))
    allowed, elsewhere = ([], {})
    for n in names:
        s = states.get(n) or {}
        if one and s.get('begun') and (not s.get('finished')) and (s.get('gpu') != key):
            elsewhere[n] = s.get('gpu')
            continue
        allowed.append(n)
    left = [n for n in allowed if not (states.get(n) or {}).get('finished')]
    return {'allowed': allowed, 'elsewhere': elsewhere, 'left': left}
