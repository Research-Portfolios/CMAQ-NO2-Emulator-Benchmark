"""The gates of the evaluation: the test-day evaluation runs only when every listed training and baseline has a verified done marker and the opening record is present, and the held-back-day evaluation also needs its own opening record; sealed files are tested by name and size only."""
import hashlib
import json
import os
import re
DECISION_ID = re.compile('^R-[0-9]+$')
SHA = re.compile('^[0-9a-f]{64}$')
PKGNAME = re.compile('^code/[A-Za-z0-9_]+$')
PLAIN = re.compile('^[A-Za-z0-9_.+-]+$')
RUNID_CV = re.compile('^eval_cv_[0-9]{8}_[0-9]{6}$')
UTC = re.compile('^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$')
FOLDS = (2015, 2016, 2017, 2018, 2019)
FAMILY = ('M', 'P', 'GBT', 'CLIM-WD')
SEALED_TOP = 'W_P2/results_sealed'
MARKER_TOPS = ('TRAIN/logs/', 'RUN/logs/')
LIST_MAX = 12

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def at(emul, rel):
    return os.path.join(emul, *rel.split('/'))

def below(rel, tops):
    if not isinstance(rel, str) or not rel or rel.startswith('/') or ('\\' in rel) or (':' in rel):
        return False
    parts = rel.split('/')
    if any((p in ('', '.', '..') for p in parts)):
        return False
    return any((rel.startswith(t) for t in tops))

def sealed_path(rel):
    return rel == SEALED_TOP or rel.startswith(SEALED_TOP + '/')

def parse_opening(raw, spec):
    errs = []
    try:
        text = raw.decode('ascii')
    except UnicodeDecodeError:
        return ({}, ['it is not 7-bit ASCII'])
    if '\r' in text:
        errs.append('it has CR characters (LF only)')
    lines = text.replace('\r', '').split('\n')
    while lines and (not lines[-1].strip()):
        lines.pop()
    if not lines or lines[0].strip() != spec['first_line']:
        errs.append("its first line is not '%s'" % spec['first_line'])
    if len(lines) < 2 or lines[-1].strip() != spec['last_line']:
        errs.append("its last line is not '%s'" % spec['last_line'])
    out = {k: [] for k in spec.get('keys_many', [])}
    for ln in lines[1:-1]:
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        if ': ' not in s:
            errs.append("a line is not '<key>: <value>'")
            continue
        k, v = s.split(': ', 1)
        k, v = (k.strip(), v.strip())
        if k in spec['keys_once']:
            if k in out:
                errs.append('%s is given twice' % k)
            out[k] = v
        elif k in spec.get('keys_many', []):
            out[k].append(v)
        else:
            errs.append("unknown key '%s'" % k[:40])
    for k in spec['keys_once']:
        if k not in out:
            errs.append('%s is missing' % k)
    if 'schema' in out and out['schema'] != spec['schema']:
        errs.append('schema is not %s' % spec['schema'])
    if 'decided_by' in out and (not DECISION_ID.match(out['decided_by'])):
        errs.append('decided_by is not a decision id (R-<digits>)')
    if 'made_utc' in out and (not UTC.match(out['made_utc'])):
        errs.append('made_utc is not yyyy-mm-ddTHH:MM:SSZ')
    for k in ('models_sha256', 'cv_opening_sha256', 'cv_eval_done_sha256'):
        if k in out and (not SHA.match(out[k])):
            errs.append('%s is not a sha256 (64 lower-case hex)' % k)
    if 'cv_run_id' in out and (not RUNID_CV.match(out['cv_run_id'])):
        errs.append('cv_run_id is not eval_cv_<yyyymmdd_HHMMSS>')
    return (out, errs)

def parse_packages(lines):
    out, errs, seen = ([], [], set())
    for v in lines:
        parts = v.split()
        if len(parts) != 2 or not PKGNAME.match(parts[0]) or (not SHA.match(parts[1])):
            errs.append("a package line is not 'code/<name> <sha256>'")
            continue
        if parts[0] in seen:
            errs.append('package %s is named twice' % parts[0])
            continue
        seen.add(parts[0])
        out.append((parts[0], parts[1]))
    return (out, errs)

def parse_nb8_end(lines, default):
    out, errs = ([], [])
    for v in lines:
        if ' :: ' not in v:
            errs.append("an nb8_end line is not '<path> :: <text>'")
            continue
        p, t = v.split(' :: ', 1)
        texts = [x.strip() for x in t.split('&&') if x.strip()]
        p = p.strip()
        if not below(p, MARKER_TOPS) or sealed_path(p) or (not texts):
            errs.append('an nb8_end path is not below TRAIN/logs/ or RUN/logs/, or names no text')
            continue
        out.append({'path': p, 'contains': texts, 'line': 'last'})
    if not lines:
        for d in default:
            e = {'path': d['path'], 'contains': list(d['contains']), 'line': 'last_run'}
            if d.get('status_file'):
                e.update(line='last_run_lane', lane=d.get('lane'), status_file=d['status_file'])
            out.append(e)
    return (out, errs)

def read_models(emul, rel, want_sha, spec):
    if not below(rel, (spec['models_under'],)):
        return (None, ["the models table's path is not below %s" % spec['models_under']])
    p = at(emul, rel)
    if not os.path.isfile(p):
        return (None, ['the models table EMUL/%s is absent' % rel])
    with open(p, 'rb') as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != want_sha:
        return (None, ['the models table EMUL/%s is not the file the opening names (sha256 differs)' % rel])
    return models_rows(raw, spec)

def models_rows(raw, spec):
    try:
        d = json.loads(raw.decode('ascii'))
    except (UnicodeDecodeError, ValueError):
        return (None, ['the models table is not ASCII JSON'])
    if not isinstance(d, dict) or d.get('schema') != spec['models_schema'] or (not isinstance(d.get('rows'), list)) or (not d['rows']):
        return (None, ['the models table is not {schema %s, rows}' % spec['models_schema']])
    rows, seen = ([], set())
    for r in d['rows']:
        if not isinstance(r, dict):
            return (None, ['the models table has a row that is not an object'])
        try:
            fold = int(r.get('fold'))
        except (TypeError, ValueError):
            fold = None
        seed = r.get('seed')
        if r.get('kind') not in ('network', 'baseline') or not isinstance(r.get('model'), str) or (not PLAIN.match(str(r.get('name', '')))) or (str(r.get('name')) in ('.', '..')) or (fold not in FOLDS) or (not (seed is None or (isinstance(seed, int) and (not isinstance(seed, bool))))):
            return (None, ['the models table has a row without model, kind, a plain name, a fold year or an integer seed'])
        key = (r['model'], fold, seed)
        if key in seen:
            return (None, ['the models table lists a model, fold and seed twice'])
        seen.add(key)
        rows.append({'model': r['model'], 'kind': r['kind'], 'name': str(r['name']), 'fold': fold, 'seed': seed})
    return (rows, [])

def _base(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]

def train_done(emul, places, name, fold, spec):
    cdir = at(emul, places['ckpt'] + '/' + name)
    p = os.path.join(cdir, spec['file'])
    if not os.path.isfile(p):
        return (False, 'DONE.json absent')
    try:
        with open(p, 'rb') as f:
            d = json.loads(f.read().decode('ascii'))
    except (OSError, UnicodeDecodeError, ValueError):
        return (False, 'DONE.json unreadable')
    if not isinstance(d, dict):
        return (False, 'DONE.json is not an object')
    keep = {k: d.get(k) for k in ('schema', 'status', 'training', 'files', 'best_pt', 'labels')}
    del d
    if not re.match(spec['schema_pattern'], str(keep['schema'])):
        return (False, 'DONE.json schema is not choi-train-done-<n>')
    if keep['status'] != spec['status'] or keep['training'] != name:
        return (False, 'DONE.json is not done or names another training')
    lab = keep['labels'] if isinstance(keep['labels'], dict) else {}
    if lab.get('fold') is not None:
        try:
            if int(lab['fold']) != int(fold):
                return (False, 'DONE.json labels another fold')
        except (TypeError, ValueError):
            return (False, "DONE.json's fold label is not a year")
    files = keep['files']
    if not isinstance(files, dict) or 'pred_masked.npy' not in files:
        return (False, 'DONE.json lists no pred_masked.npy')
    bp = keep['best_pt']
    if not isinstance(bp, dict) or not bp.get('path'):
        return (False, 'DONE.json has no best.pt row')
    pdir = at(emul, places[spec['pred_place']] + '/' + name)
    rows = [(pdir, r) for r in files.values()] + [(cdir, bp)]
    for folder, r in rows:
        if not isinstance(r, dict):
            return (False, 'a DONE.json file row is not an object')
        b = _base(r.get('path', ''))
        if not b or b in ('.', '..'):
            return (False, 'a DONE.json file row has no name')
        fp = os.path.join(folder, b)
        try:
            ok = os.path.isfile(fp) and os.stat(fp).st_size == int(r.get('bytes', -1))
        except (TypeError, ValueError, OSError):
            ok = False
        if not ok:
            return (False, '%s absent or not of its listed bytes' % b)
    return (True, 'verified')

def cpu_done(emul, places, spec, baselines):
    p = at(emul, spec['file'])
    if not os.path.isfile(p):
        return ['(3) EMUL/%s is absent: the CPU stage has not finished' % spec['file']]
    with open(p, 'rb') as f:
        lines = f.read().decode('ascii', 'replace').splitlines()
    first = lines[0] if lines else ''
    last = lines[-1] if lines else ''
    if not (first.startswith(spec['first_line_starts']) and spec['first_line_has'] in first):
        return ["(3) EMUL/%s's first line is not '%s ... %s': the CPU stage has not finished" % (spec['file'], spec['first_line_starts'], spec['first_line_has'])]
    if not last.startswith(spec['last_line_starts']):
        return ['(3) EMUL/%s does not end with its END line' % spec['file']]
    listed = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) == 4 and parts[0] == 'FILE' and (parts[1] == 'CV_DONE'):
            try:
                listed[parts[2]] = int(parts[3])
            except ValueError:
                pass
    miss = []
    for r in baselines:
        rel = '%s/fold%d/DONE.json' % (r['name'], r['fold'])
        if rel not in listed:
            miss.append('(3) baseline %s fold %d: CPU_DONE lists no CV_DONE row' % (r['name'], r['fold']))
            continue
        fp = at(emul, places['baselines_sealed'] + '/' + rel)
        if not (os.path.isfile(fp) and os.stat(fp).st_size == listed[rel]):
            miss.append('(3) baseline %s fold %d: its DONE.json is absent or not of the bytes CPU_DONE lists' % (r['name'], r['fold']))
    return miss
NOT_REACHED = re.compile('not reached ([0-9]+) of ([0-9]+)')

def status_not_reached(emul, rel):
    if not below(rel, MARKER_TOPS) or sealed_path(rel):
        return None
    p = at(emul, rel)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, 'rb') as f:
            d = json.loads(f.read().decode('ascii'))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    res = d.get('results') if isinstance(d, dict) else None
    del d
    if not isinstance(res, dict):
        return None
    return sorted((str(k) for k, v in res.items() if str(v).startswith('not reached')))

def nb8_end(emul, entries, rows=None):
    miss = []
    table = set((r['name'] for r in rows or [] if r['kind'] == 'network'))
    for e in entries:
        lane = 'lane %s: ' % e['lane'] if e.get('lane') is not None else ''
        p = at(emul, e['path'])
        if not os.path.isfile(p):
            miss.append("(4) %snotebook 8's end marker EMUL/%s is absent" % (lane, e['path']))
            continue
        with open(p, 'rb') as f:
            lines = [x for x in f.read().decode('ascii', 'replace').splitlines() if x.strip()]
        if e['line'] in ('last_run', 'last_run_lane'):
            lines = [x for x in lines if x.lstrip().startswith('run: ')]
        last = lines[-1] if lines else ''
        if not all((t in last for t in e['contains'])):
            miss.append("(4) %sEMUL/%s does not show the training stage's end (its last %sline lacks '%s')" % (lane, e['path'], "'run: ' " if e['line'] in ('last_run', 'last_run_lane') else '', ' && '.join(e['contains'])))
            continue
        if e['line'] != 'last_run_lane':
            continue
        m = NOT_REACHED.search(last)
        if m is None:
            miss.append("(4) %sEMUL/%s: its last 'run: ' line has no 'not reached <n> of <n>'" % (lane, e['path']))
            continue
        n = int(m.group(1))
        if n == 0:
            continue
        names = status_not_reached(emul, e['status_file'])
        if names is None or len(names) != n:
            miss.append("(4) %snotebook 8's last run left %d trainings not reached, and EMUL/%s does not name them" % (lane, n, e['status_file']))
            continue
        if rows is None:
            miss.append('(4) %s%d trainings not reached: not checked (no readable models table)' % (lane, n))
            continue
        for name in names:
            if name in table:
                miss.append('(4) %straining %s is in the models table but the training stage did not reach it (EMUL/%s)' % (lane, name, e['path']))
    return miss

def check_cv(emul, conf, places, own_sums_sha):
    g = conf['nb9']['gate_cv']
    roots = conf['nb9']['code_root']
    st = {'opening': None, 'opening_sha256': None, 'models': None, 'models_sha256': None, 'packages': []}
    miss = []
    op = at(emul, g['opening_file'])
    fields = None
    if not os.path.isfile(op):
        miss.append("(1) the user's opening file EMUL/%s is absent (the list of trainings and items (2) and (4) are read from it)" % g['opening_file'])
    else:
        with open(op, 'rb') as f:
            raw = f.read()
        st['opening_sha256'] = hashlib.sha256(raw).hexdigest()
        fields, errs = parse_opening(raw, g)
        pk, perr = parse_packages(fields.get('package', []))
        ends, eerr = parse_nb8_end(fields.get('nb8_end', []), g['nb8_end_default'])
        errs += perr + eerr
        names = [n for n, _s in pk]
        for need in g['packages_required']:
            if need not in names:
                errs.append('it names no package line for %s' % need)
        want = {roots['base']: roots['base_sums_sha256'], roots['p7']: roots['p7_sums_sha256'], conf['nb9']['package']: own_sums_sha}
        for n, s in pk:
            if n in want and s != want[n]:
                errs.append('its package %s is not the version this notebook runs (sums sha256 differs)' % n)
                continue
            sp = at(emul, n + '/' + g['package_sums_file'])
            if not os.path.isfile(sp):
                errs.append('package %s has no %s on the Drive' % (n, g['package_sums_file']))
            elif sha256_file(sp) != s:
                errs.append("package %s's sums file on the Drive is not the one the opening names" % n)
        for e in errs:
            miss.append('(1) the opening file EMUL/%s: %s' % (g['opening_file'], e))
        if not errs:
            st['opening'] = fields
            st['packages'] = pk
            st['nb8_end'] = ends
    rows = None
    if st['opening'] is not None:
        rows, merr = read_models(emul, fields['models'], fields['models_sha256'], g)
        for e in merr:
            miss.append('(1) %s' % e)
        if rows is not None:
            st['models'] = rows
            st['models_sha256'] = fields['models_sha256']
            have = set(((r['model'], r['fold']) for r in rows if r['seed'] in (0, None)))
            for m in FAMILY:
                for y in FOLDS:
                    if (m, y) not in have:
                        miss.append('(1) the models table lists no %s (its s0 member) on fold %d: the family is read on every fold at once' % (m, y))
    if rows is None:
        miss.append("(2) the trainings' DONE markers: not checked (no readable list of trainings)")
    else:
        seen, bad = (set(), [])
        for r in rows:
            if r['kind'] != 'network' or r['name'] in seen:
                continue
            seen.add(r['name'])
            ok, why = train_done(emul, places, r['name'], r['fold'], g['train_done'])
            if not ok:
                bad.append('%s (%s)' % (r['name'], why))
        for b in bad[:LIST_MAX]:
            miss.append('(2) training %s: no verified DONE marker' % b)
        if len(bad) > LIST_MAX:
            miss.append('(2) and %d more trainings without a verified DONE marker' % (len(bad) - LIST_MAX))
    base_rows = [r for r in rows or [] if r['kind'] == 'baseline']
    cm = cpu_done(emul, places, g['cpu_done'], base_rows)
    miss += cm[:LIST_MAX]
    if len(cm) > LIST_MAX:
        miss.append('(3) and %d more baseline lines' % (len(cm) - LIST_MAX))
    if st['opening'] is not None:
        miss += nb8_end(emul, st['nb8_end'], rows)
    else:
        miss.append("(4) the training stage's end marker: not checked (no readable opening file; the default is %s)" % ' and '.join(('EMUL/%s' % d['path'] for d in g['nb8_end_default'])))
    if not os.path.isfile(at(emul, g['pd17_record'])):
        miss.append("(5) the study design's record EMUL/%s is absent (the CPU stage's pd17 step)" % g['pd17_record'])
    return (st, miss)

def cv_marker_rel(conf, opening_sha):
    return '%s/EVAL_CV_DONE_%s_ASCII.txt' % (conf['nb9']['logs'], opening_sha[:16])

def lockbox_marker_rel(conf):
    return '%s/EVAL_LOCKBOX_DONE_ASCII.txt' % conf['nb9']['logs']

def read_marker(p, head):
    if not os.path.isfile(p):
        return None
    with open(p, 'rb') as f:
        lines = f.read().decode('ascii', 'replace').splitlines()
    if not lines or not lines[0].startswith(head + ' ') or (not lines[0].endswith(' exit 0')):
        return None
    out = {}
    for ln in lines[1:]:
        if ': ' in ln:
            k, v = ln.split(': ', 1)
            out[k.strip()] = v.strip()
    return out

def check_lockbox(emul, conf, places, cv_state):
    g = conf['nb9']['gate_lockbox']
    p = at(emul, g['opening_file'])
    if not os.path.isfile(p):
        return (None, ["the user's lockbox opening file EMUL/%s is absent: the lockbox stage waits for a decision after the cv stage" % g['opening_file']])
    with open(p, 'rb') as f:
        raw = f.read()
    fields, errs = parse_opening(raw, g)
    miss = ['the lockbox opening file EMUL/%s: %s' % (g['opening_file'], e) for e in errs]
    if miss:
        return (None, miss)
    if fields['cv_opening_sha256'] != cv_state['opening_sha256']:
        miss.append('the lockbox opening file follows another cv opening file (cv_opening_sha256 is not the sha256 of EMUL/%s)' % conf['nb9']['gate_cv']['opening_file'])
        return (None, miss)
    mk = read_marker(at(emul, cv_marker_rel(conf, cv_state['opening_sha256'])), 'NB9 CV DONE')
    if mk is None:
        miss.append("this notebook's cv stage has not finished for this opening (EMUL/%s absent)" % cv_marker_rel(conf, cv_state['opening_sha256']))
    elif mk.get('run_id') != fields['cv_run_id'] or mk.get('eval_done_sha256') != fields['cv_eval_done_sha256']:
        miss.append("the lockbox opening file names another cv run (its cv_run_id or cv_eval_done_sha256 is not the cv marker's)")
    else:
        ed = at(emul, places['eval_sealed'] + '/' + fields['cv_run_id'] + '/EVAL_DONE.json')
        if not os.path.isfile(ed):
            miss.append("the cv run's EVAL_DONE.json is absent")
        elif sha256_file(ed) != fields['cv_eval_done_sha256']:
            miss.append("the cv run's EVAL_DONE.json is not the file the lockbox opening file names (sha256 differs)")
    return (fields if not miss else None, miss)

def pe_opening_record(conf, fields):
    d = dict(conf['nb9']['gate_lockbox']['pe_opening'])
    d['decided_by'] = fields['decided_by']
    return d
