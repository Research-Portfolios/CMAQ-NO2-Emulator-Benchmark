"""Runs the extraction of the 2015, 2016 and 2018 emission archives: plans each year's members from the tar headers, checks each archive, extracts the daily means member by member with manifest rows, and, when a year is complete, runs the year's checks and the separate value check."""
import argparse
import csv
import datetime
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import c7a_common as C
import c7a_tar as T
import c7a_reduce as R
tario, stacks, s3io = C.use_vendor()
MANIFEST_FIELDS = ['path', 'kind', 'sector', 'date', 'bytes', 'sha256', 'tar', 'tar_sha256', 'member', 'member_offset', 'member_size', 'member_sha256', 'member_sha256_kind', 'species', 'units', 'hours', 'version', 'n_stacks', 'n_summed', 'n_outside', 'base', 'p1', 'written_at']
NOWIN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
T2_BASE_NOTE = os.path.join(C.PKG, 'T2_BASELINE_NOTE_ASCII.txt')

def note_name(path):
    p = os.path.abspath(path)
    top = C.ROOT + '\\'
    return p[len(top):] if p.lower().startswith(top.lower()) else p

def t2_baseline(start, note=None, now=None):
    try:
        text = C.read_bytes(note or T2_BASE_NOTE).decode('ascii')
        base = C.parse_stamp(text.splitlines()[0])
    except (OSError, UnicodeDecodeError, IndexError, ValueError):
        return None
    if base > (now or C.now()):
        return None
    return max(start, base)

def pins_of(year, pins_fmt=C.PINS, check=True):
    p = pins_fmt % year
    raw = C.read_bytes(p)
    import hashlib
    s16 = hashlib.sha256(raw).hexdigest()[:16]
    if check and s16 != C.PINS_SHA16[year]:
        raise RuntimeError('pins_%s.csv is %s, not the decided %s' % (year, s16, C.PINS_SHA16[year]))
    rows = list(csv.DictReader(io.StringIO(raw.decode('ascii'))))
    return ({r['name']: r for r in rows}, s16)

def k4_pass(year, pins, h_input, tables):
    out = {}
    for fmt in tables:
        p = fmt % year
        if not os.path.exists(p):
            continue
        with open(p, newline='', encoding='ascii', errors='replace') as fh:
            text = fh.read()
        for r in csv.DictReader(io.StringIO(text)):
            if r.get('k4') != 'PASS' or None in r or None in r.values():
                continue
            name = os.path.basename(r.get('path', ''))
            pin = pins.get(name)
            if pin is None:
                continue
            want = os.path.join(h_input, year, 'emis', name)
            if os.path.normcase(r['path']) != os.path.normcase(want):
                continue
            if str(r.get('bytes')) != str(pin['bytes']) or r.get('md5') != pin['md5'] or r.get('sha256') != pin['sha256']:
                continue
            out[name] = r
    return out

def ref_sizes(name, plan):
    fam, mm = C.family_month(name)
    out = []
    for y in ('2017', '2019'):
        for k, v in plan.get('keys', {}).items():
            base = k.split('/')[-1]
            if not base.endswith('.tar') or '/%s/emis/' % y not in k:
                continue
            f2, m2 = C.family_month(base)
            if f2 == fam and m2 == mm:
                out.append(int(v['size']))
    return out

class Run:

    def __init__(self, years, p1, out_root=C.OUT_ROOT, h_input=C.H_INPUT, run_dir=C.RUN_DIR, k4_tables=C.K4_TABLES, pins_fmt=C.PINS, xfer_out=C.XFER_OUT, t2=None, mem=None, sleep=time.sleep, clock=None, check_pins=True, valuecheck=None, plan=None):
        self.years = list(years)
        self.p1 = bool(p1)
        self.out_root, self.h_input, self.run_dir = (out_root, h_input, run_dir)
        self.k4_tables, self.pins_fmt, self.xfer_out = (k4_tables, pins_fmt, xfer_out)
        self.t2 = t2 or self._t2
        self.mem = mem or C.mem_free_gb
        self.sleep = sleep
        self.clock = clock or C.now
        self.check_pins = check_pins
        self.valuecheck = valuecheck or self._valuecheck
        self.plan = plan if plan is not None else C.read_json(C.S2A_PLAN, {})
        self.once = False
        sp = os.path.join(self.run_dir, 'RUN_START_ASCII.txt')
        if os.path.exists(sp):
            self.start = C.parse_stamp(C.read_bytes(sp).decode('ascii'))
        else:
            self.start = self.clock()
            if self.years:
                os.makedirs(self.run_dir, exist_ok=True)
                with open(sp, 'w', encoding='ascii', newline='\n') as f:
                    f.write(C.stamp(self.start) + '\n')
        self.t2_last = None
        self.t2_fails = 0
        self.cf_fired = False
        self.t3_alerted = False
        self.groups = {}
        self.timings = []
        self.stop_reason = None
        self.code_sha16 = {f: C.sha16(os.path.join(C.CODE, *f.split('\\'))) for f in ('c7a_common.py', 'c7a_tar.py', 'c7a_reduce.py', 'c7a_extract.py', 'ncclassic.py', 'c7a_valuecheck.py', 'vendor\\PINS_ASCII.txt', 'vendor\\choi\\tario.py', 'vendor\\choi\\stacks.py', 'vendor\\choi_dl\\s3io.py')}

    def log(self, text):
        C.append_record(os.path.join(self.run_dir, 'RUN_LOG_ASCII.txt'), '%s %s' % (C.stamp(self.clock()), text))

    def alert(self, step, text):
        C.append_record(os.path.join(self.run_dir, 'ALERTS_ASCII.txt'), '[%s] %s: %s' % (C.stamp(self.clock()), step, text))
        self.log('ALERT %s: %s' % (step, text))

    def ydir(self, y):
        return os.path.join(self.out_root, y)

    def state(self, y):
        return C.read_json(os.path.join(self.ydir(y), 'STATE_%s.json' % y), {'year': y, 'tars': {}, 'groups': {}, 'issues': [], 'held': {}, 'year_status': None})

    def save_state(self, y, st):
        st['p1'] = 'on' if self.p1 else 'off'
        st['code_sha16'] = self.code_sha16
        C.write_json(os.path.join(self.ydir(y), 'STATE_%s.json' % y), st)
        C.write_json(os.path.join(self.run_dir, 'STATE_%s.json' % y), st)

    def manifest(self, y):
        p = os.path.join(self.ydir(y), 'MANIFEST_%s.csv' % y)
        if not os.path.exists(p):
            return {}
        with open(p, newline='', encoding='ascii') as fh:
            rows = list(csv.DictReader(fh))
        return {(r['kind'], r['sector'], r['date']): r for r in rows}

    def manifest_rows(self, y):
        p = os.path.join(self.ydir(y), 'MANIFEST_%s.csv' % y)
        if not os.path.exists(p):
            return []
        with open(p, newline='', encoding='ascii') as fh:
            return list(csv.DictReader(fh))

    def add_manifest(self, y, row):
        for p in (os.path.join(self.ydir(y), 'MANIFEST_%s.csv' % y), os.path.join(self.run_dir, 'MANIFEST_%s.csv' % y)):
            new = not os.path.exists(p)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'a', newline='', encoding='ascii') as fh:
                w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator='\n')
                if new:
                    w.writeheader()
                w.writerow(row)

    def _t2(self):
        if C.sha16(C.T2_SCRIPT) != C.T2_SHA16:
            return (2, 'wait_t2.py changed (sha256 first 16)')
        base = t2_baseline(self.start, now=self.clock())
        if base is None:
            return (2, 'T-2 baseline note %s missing, unreadable or later than now; T-2 not read' % note_name(T2_BASE_NOTE))
        p = subprocess.run([C.VENV_PY, '-I', '-B', C.T2_SCRIPT, base.astimezone().strftime('%Y-%m-%d %H:%M:%S')], capture_output=True, timeout=300, creationflags=NOWIN)
        lines = [ln.strip() for ln in p.stdout.decode('ascii', 'replace').splitlines()]
        line = next((ln for ln in lines if ln), '')
        return (p.returncode, C.mask(line)[:300])

    def gate_t2(self, force=False):
        now = self.clock()
        if not force and self.t2_last and ((now - self.t2_last).total_seconds() < C.T2_EVERY_S):
            return 'go'
        self.t2_last = now
        if t2_baseline(self.start, now=now) is None:
            self.alert('T-2', 'the T-2 baseline note %s is missing, unreadable or later than now; the pass stops before T-2 is read and before any further tar; the user writes the note' % note_name(T2_BASE_NOTE))
            return 'T-2 baseline note'
        try:
            rc, line = self.t2()
        except Exception as e:
            rc, line = (2, 'T-2 read raised %s' % type(e).__name__)
        cnt = os.path.join(self.run_dir, 'T2_FAILS_ASCII.txt')
        if rc == 0:
            self.t2_fails = 0
            if os.path.exists(cnt):
                os.remove(cnt)
            self.log('T-2 read: no event (%s)' % line)
            return 'go'
        if rc == 1:
            self.alert('T-2', 'EVENT ON the archive disk: %s; all writes to the archive disk stop; route B only by a manual decision' % line)
            return 'T-2'
        prev = 0
        if os.path.exists(cnt):
            try:
                prev = int(C.read_bytes(cnt).decode('ascii').strip())
            except ValueError:
                prev = 0
        self.t2_fails = max(self.t2_fails, prev) + 1
        os.makedirs(self.run_dir, exist_ok=True)
        with open(cnt, 'w', encoding='ascii', newline='\n') as f:
            f.write('%d\n' % self.t2_fails)
        self.alert('T-2', 'NOT READ (%d in a row): %s' % (self.t2_fails, line))
        if self.t2_fails >= C.T2_FAILS_STOP:
            return 'T-2 not read'
        return 'go'

    def gates(self):
        if os.path.exists(C.STOP_FILE):
            return 'stop file'
        g = self.gate_t2()
        if g != 'go':
            return g
        need = C.MEM_FLOOR_GB + C.PEAK_GB
        waited = False
        while self.mem() < need:
            if self.once:
                self.log('memory %.2f GB under %.1f GB: the pass ends at this tar boundary; the watcher waits' % (self.mem(), need))
                return 'memory'
            if not waited:
                self.log('memory %.2f GB under %.1f GB (4.6 + peak %.1f); waiting in %d s steps' % (self.mem(), need, C.PEAK_GB, C.MEM_WAIT_S))
                waited = True
            self.cf_check()
            if self.clock() >= C.parse_stamp(C.C7A_DEADLINE):
                return 'C-7a (1) deadline'
            for _ in range(max(1, C.MEM_WAIT_S // 10)):
                if os.path.exists(C.STOP_FILE):
                    return 'stop file'
                self.sleep(min(10, C.MEM_WAIT_S))
        if waited:
            self.log('memory %.2f GB: go' % self.mem())
        return 'go'

    def tar_path(self, y, name):
        return os.path.join(self.h_input, y, 'emis', name)

    def process_tar(self, y, name, pins, st):
        t0 = time.time()
        path = self.tar_path(y, name)
        cname = name[:-4] + '.contents.txt'
        info = {'status': 'failed', 'bytes_pinned': int(pins[name]['bytes']), 'tar_sha256': pins[name]['sha256']}
        info['k2'] = T.k2(path, pins[name]['bytes'], ref_sizes(name, self.plan))
        try:
            members, w = T.walk(path)
            cbytes = C.read_bytes(self.tar_path(y, cname))
            info['k3'] = T.k3(members, cbytes)
        except (tario.TarHeaderError, OSError) as e:
            info['k3'] = {'ok': False, 'error': C.mask(str(e))[:200]}
            members = []
        if info['k2']['flag']:
            self.alert('K2 FLAG', '%s: %d bytes beside 2017 and 2019 %s' % (name, info['k2']['bytes'], info['k2']['ref_2017_2019']))
        if not (info['k2']['ok'] and info['k3']['ok']):
            self.alert('K2/K3', '%s FAILED: K2 %s, K3 %s' % (name, info['k2']['ok'], json.dumps(info['k3'])[:300]))
            st['tars'][name] = info
            return info
        need, issues = T.pick(members, y)
        st['issues'].extend(('%s: %s' % (name, i) for i in issues))
        kind = C.kind_of_tar(name)
        n = 0
        with open(path, 'rb') as fh:
            if kind == 'stackgroups':
                n = self.do_groups(y, need, fh, st)
            else:
                man = self.manifest(y)
                for key in sorted(need, key=lambda k: (k[0], k[1] or '', k[2] or '')):
                    k, s, d = key
                    if kind == 'merged' and k == 'merged':
                        n += self.do_member(y, key, need[key], fh, name, info['tar_sha256'], man, st)
                    elif kind == 'ptsectors' and k == 'pt':
                        if s in st['held']:
                            continue
                        n += self.do_member(y, key, need[key], fh, name, info['tar_sha256'], man, st)
        info.update(status='done', members_done=n, seconds=round(time.time() - t0, 1), done_at=C.stamp(self.clock()))
        st['tars'][name] = info
        self.timings.append((kind, info['bytes_pinned'], info['seconds']))
        return info

    def do_groups(self, y, need, fh, st):
        n = 0
        for s in C.SECTORS:
            m = need.get(('groups', s, None))
            if m is None:
                st['held'][s] = 'no stack groups file in the tar'
                self.alert('GROUPS', '%s %s: no stack groups file; the sector is held for the year' % (y, s))
                continue
            ext = m['name'].rsplit('.', 1)[-1]
            try:
                g = R.read_groups(fh, m, ext)
            except Exception as e:
                g = {'reason': 'read error %s: %s' % (type(e).__name__, C.mask(str(e))[:120])}
            rec = {'member': m['name'], 'n_stacks': g.get('n_stacks'), 'reason': g.get('reason'), 'member_sha256': g.get('member_sha256')}
            if g.get('reason') or 'latitude' not in g:
                st['held'][s] = 'groups: %s' % (g.get('reason') or 'no LATITUDE')
                self.alert('GROUPS', '%s %s: %s' % (y, s, st['held'][s]))
            else:
                rc = R.rowcol_check(g, C.GRID_ATTRS)
                rec['rowcol'] = rc
                if rc['base'] != C.BASE_EXPECTED:
                    st['held'][s] = 'ROW and COL base %d, not %d' % (rc['base'], C.BASE_EXPECTED)
                    self.alert('GROUPS', '%s %s: %s' % (y, s, st['held'][s]))
                self.groups[y, s] = g
                n += 1
            st['groups'][s] = rec
        return n

    def ensure_groups(self, y, st):
        need = [s for s in C.SECTORS if (y, s) not in self.groups and s not in st['held']]
        if not need or not st['groups']:
            return
        sg = [t for t in st['tars'] if C.kind_of_tar(t) == 'stackgroups' and st['tars'][t]['status'] == 'done']
        if not sg:
            return
        members, w = T.walk(self.tar_path(y, sg[0]))
        found, _ = T.pick(members, y)
        with open(self.tar_path(y, sg[0]), 'rb') as fh:
            for s in need:
                m = found.get(('groups', s, None))
                if m is not None:
                    self.groups[y, s] = R.read_groups(fh, m, m['name'].rsplit('.', 1)[-1])

    def do_member(self, y, key, m, fh, tar, tar_sha, man, st):
        k, s, d = key
        rel = os.path.join('emis_merged', 'emis_merged_%s.npz' % d) if k == 'merged' else os.path.join('pt_marine', s, 'pt_%s_%s.npz' % (s, d))
        path = os.path.join(self.ydir(y), rel)
        old = man.get((k, s or '', d))
        if old and os.path.exists(path) and (C.sha256_file(path)[0] == old['sha256']):
            return 1
        ext = m['name'].rsplit('.', 1)[-1]
        kind = 'merged' if k == 'merged' else 'stack'
        try:
            res = R.read_selective(fh, m, ext, kind)
        except Exception as e:
            why = 'read error %s: %s' % (type(e).__name__, C.mask(str(e))[:120])
            st['issues'].append('MEMBER %s %s %s %s: %s' % (tar, k, s or '', d, why))
            self.alert('MEMBER', '%s %s %s %s: %s' % (y, k, s or '', d, why))
            return 0
        hc = res['hours']
        bad = []
        if k == 'pt' and (res.get('n_stacks') or 0) > C.PEAK_STACKS:
            self.alert('PEAK', '%s %s %s: %s stacks, more than the %d the peak was measured at; the user re-measures' % (y, s, d, res['n_stacks'], C.PEAK_STACKS))
        if not hc.get('ok') or hc.get('date') != C.yyyyddd(d):
            bad.append('hours: %s (TFLAG date %s, expected %d)' % (hc.get('reason'), hc.get('date'), C.yyyyddd(d)))
        sp_list = C.MERGED_SPECIES if k == 'merged' else C.PT_SPECIES
        miss = [sp for sp in sp_list if sp not in res['means']]
        if miss:
            bad.append('species missing: %s' % ' '.join(miss))
        units = [str(res['units'].get(sp)) for sp in sp_list]
        for sp, u in zip(sp_list, units):
            if sp in res['means'] and u not in C.UNITS_OK[sp]:
                bad.append('units %s: %s' % (sp, u))
        arrays = {}
        info = {}
        if k == 'merged':
            at = res.get('attrs', {})
            for a, v in C.GRID_ATTRS.items():
                if a in at and float(at[a]) != float(v):
                    bad.append('grid attribute %s differs' % a)
            for sp in sp_list:
                a = res['means'].get(sp)
                if a is not None and (a.shape != C.GRID or a.dtype != np.float32):
                    bad.append('%s shape %s dtype %s' % (sp, a.shape, a.dtype))
            if not bad:
                arrays = {sp: res['means'][sp] for sp in sp_list}
        elif not bad:
            if (y, s) not in self.groups:
                self.ensure_groups(y, st)
            g = self.groups.get((y, s))
            if g is None:
                st['held'][s] = 'no groups read'
                bad.append('no groups for %s' % s)
            elif res['n_stacks'] != g['n_stacks']:
                st['held'][s] = 'stack count mismatch: file %s, groups %s (a manual decision is needed)' % (res['n_stacks'], g['n_stacks'])
                self.alert('STACKS', '%s %s %s: %s' % (y, s, d, st['held'][s]))
                bad.append('stack count mismatch')
            else:
                grids, info = R.cell_sums(res, g, C.BASE_EXPECTED, self.p1)
                arrays = grids
        if bad:
            st['issues'].append('MEMBER %s %s %s %s: %s' % (tar, k, s or '', d, '; '.join(bad)))
            self.alert('MEMBER', '%s %s %s %s: %s' % (y, k, s or '', d, '; '.join(bad)))
            return 0
        meta = {'species': np.array(sp_list), 'units': np.array(units), 'hours': np.array('00-23 UTC'), 'date': np.array(d), 'member': np.array(m['name']), 'tar': np.array(tar), 'tar_sha256': np.array(tar_sha), 'member_offset': np.array(int(m['data_offset']), dtype=np.int64), 'member_size': np.array(int(m['size']), dtype=np.int64), 'member_sha256': np.array(res['member_sha256']), 'member_sha256_kind': np.array(res['member_sha256_kind']), 'reader': np.array(res['reader']), 'tflag_date': np.array(hc['date'], dtype=np.int64), 'p1': np.array('on' if self.p1 else 'off')}
        if k == 'pt':
            meta.update(sector=np.array(s), n_stacks=np.array(info['n_stacks'], dtype=np.int64), n_summed=np.array(info['n_summed'], dtype=np.int64), n_outside=np.array(info['n_outside'], dtype=np.int64), base=np.array(C.BASE_EXPECTED, dtype=np.int64))
        arrays.update(meta)
        nb, sha = R.write_npz(path, arrays)
        self.add_manifest(y, {'path': rel, 'kind': k, 'sector': s or '', 'date': d, 'bytes': nb, 'sha256': sha, 'tar': tar, 'tar_sha256': tar_sha, 'member': m['name'], 'member_offset': int(m['data_offset']), 'member_size': int(m['size']), 'member_sha256': res['member_sha256'], 'member_sha256_kind': res['member_sha256_kind'], 'species': ' '.join(sp_list), 'units': ' '.join(units), 'hours': '00-23 UTC', 'version': 'EQUATES v1.0', 'n_stacks': info.get('n_stacks', ''), 'n_summed': info.get('n_summed', ''), 'n_outside': info.get('n_outside', ''), 'base': C.BASE_EXPECTED if k == 'pt' else '', 'p1': 'on' if self.p1 else 'off', 'written_at': C.stamp(self.clock())})
        man[k, s or '', d] = {'sha256': sha}
        return 1

    def k_checks(self, y, pins, passed):
        st = self.state(y)
        out = {}
        names = sorted(pins)
        k1_missing = [n for n in names if n not in passed]
        k1_size = [n for n in names if n in passed and (not os.path.exists(self.tar_path(y, n)) or os.path.getsize(self.tar_path(y, n)) != int(pins[n]['bytes']))]
        out['K1'] = {'ok': len(names) == 50 and (not k1_missing) and (not k1_size), 'n_pinned': len(names), 'no_k4_pass': k1_missing, 'size_differs': k1_size}
        tars = [n for n in names if n.endswith('.tar')]
        out['K2'] = {'ok': all((st['tars'].get(t, {}).get('k2', {}).get('ok') for t in tars)), 'flags': [t for t in tars if st['tars'].get(t, {}).get('k2', {}).get('flag')]}
        out['K3'] = {'ok': all((st['tars'].get(t, {}).get('k3', {}).get('ok') for t in tars))}
        fields = ('path', 'source_link', 'date_saved', 'bytes', 'sha256')
        k4bad = [n for n in names if n in passed and (any((not passed[n].get(f) for f in fields)) or passed[n].get('version') != 'EQUATES v1.0')]
        out['K4'] = {'ok': not k1_missing and (not k4bad), 'bad': k4bad}
        out['K5'] = self.k5(y, st)
        out['K6'] = self.k6(y, st)
        return out

    def find(self, y, tar_kind, mm, key):
        for name in os.listdir(os.path.join(self.h_input, y, 'emis')):
            if C.kind_of_tar(name) == tar_kind and C.family_month(name)[1] == mm and name.endswith('.tar'):
                ms, w = T.walk(self.tar_path(y, name))
                found, _ = T.pick(ms, y)
                return (name, found.get(key))
        return (None, None)

    def k5(self, y, st):
        d = y + C.K5_MMDD
        out = {'ok': False, 'day': d}
        tm, m = self.find(y, 'merged', d[4:6], ('merged', None, d))
        tp, mp = self.find(y, 'ptsectors', d[4:6], ('pt', 'ptegu', d))
        if m is None or mp is None:
            out['why'] = 'the 1 July merged or ptegu member not found'
            return out
        with open(self.tar_path(y, tm), 'rb') as fh:
            sel = R.read_selective(fh, m, m['name'][-3:], 'merged')
            who = R.read_whole(fh, m, m['name'][-3:], 'merged')
        eq = {}
        for sp in C.MERGED_SPECIES:
            a, b = (sel['means'].get(sp), who['means'].get(sp))
            eq[sp] = a is not None and b is not None and (a.shape == b.shape) and (int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))) == 0)
        ext_ok = self._extract_equal(y, 'emis_merged', 'emis_merged_%s.npz' % d, sel['means'])
        out['merged'] = {'bitwise_equal': eq, 'units_equal': sel['units'] == who['units'], 'hours_equal': sel['hours'] == who['hours'], 'extract_equal': ext_ok, 'whole_sha256': who['member_sha256']}
        del sel, who
        with open(self.tar_path(y, tp), 'rb') as fh:
            sel = R.read_selective(fh, mp, mp['name'][-3:], 'stack')
            who = R.read_whole(fh, mp, mp['name'][-3:], 'stack')
        eqp = {}
        for sp in C.PT_SPECIES:
            a, b = (sel['means'].get(sp), who['means'].get(sp))
            eqp[sp] = a is not None and b is not None and (a.shape == b.shape) and (int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64))) == 0)
        g = self.groups.get((y, 'ptegu'))
        if g is None:
            self.ensure_groups(y, st)
            g = self.groups.get((y, 'ptegu'))
        cells = None
        if g is not None and sel['n_stacks'] == g['n_stacks']:
            gs, _ = R.cell_sums(sel, g, C.BASE_EXPECTED, self.p1)
            gw, _ = R.cell_sums(who, g, C.BASE_EXPECTED, self.p1)
            cells = all((np.array_equal(gs[k].view(np.uint32), gw[k].view(np.uint32)) for k in gs))
            ext_p = self._extract_equal(y, os.path.join('pt_marine', 'ptegu'), 'pt_ptegu_%s.npz' % d, gs)
        else:
            ext_p = False
        out['ptegu'] = {'bitwise_equal': eqp, 'units_equal': sel['units'] == who['units'], 'hours_equal': sel['hours'] == who['hours'], 'n_stacks_equal': sel['n_stacks'] == who['n_stacks'], 'cell_grids_equal': cells, 'extract_equal': ext_p, 'whole_sha256': who['member_sha256']}
        out['ok'] = all(eq.values()) and out['merged']['units_equal'] and out['merged']['hours_equal'] and ext_ok and all(eqp.values()) and out['ptegu']['units_equal'] and out['ptegu']['hours_equal'] and out['ptegu']['n_stacks_equal'] and bool(cells) and ext_p
        return out

    def _extract_equal(self, y, sub, fname, arrays):
        p = os.path.join(self.ydir(y), sub, fname)
        if not os.path.exists(p):
            return False
        with np.load(p, allow_pickle=False) as zf:
            z = {k: zf[k] for k in zf.files}
        for k, a in arrays.items():
            a32 = np.asarray(a, dtype=np.float32)
            if k not in z or z[k].shape != a32.shape or (not np.array_equal(z[k].view(np.uint32), a32.view(np.uint32))):
                return False
        return True

    def k6(self, y, st):
        man = self.manifest(y)
        days = C.days_of_year(y)
        out = {'ok': True, 'missing': {}, 'rep_months': {}, 'units_bad': [], 'rep_repeated': {}, 'rep_k6_1': {}, 'member_issues': len([i for i in st['issues'] if i.startswith('MEMBER')]), 'held': dict(st['held'])}
        got = {}
        for (k, s, d), r in man.items():
            got.setdefault((k, s), set()).add(d)
            sp = ('NO NO2 HONO VOC_INV' if k == 'merged' else 'NO NO2 HONO').split()
            for a, u in zip(sp, r['units'].split()):
                if u not in C.UNITS_OK[a]:
                    out['units_bad'].append('%s %s %s' % (k, s, d))
        for k, s in [('merged', '')] + [('pt', x) for x in C.DAILY_SECTORS]:
            miss = [d for d in days if d not in got.get((k, s), set())]
            if miss:
                out['missing']['%s %s' % (k, s)] = [len(miss)] + miss[:5]
        per = {}
        for r in self.manifest_rows(y):
            if r['kind'] == 'pt' and r['sector'] in C.REP_SECTORS:
                per.setdefault((r['sector'], r['date'][4:6]), []).append(r['date'])
        for s in C.REP_SECTORS:
            for m in ['%02d' % i for i in range(1, 13)]:
                dates = per.get((s, m), [])
                ok, _, rep, beyond = C.rep_month_ok(dates)
                if not ok:
                    out['rep_months'].setdefault(s, {})[m] = len(dates)
                    if rep:
                        out['rep_repeated'].setdefault(s, {})[m] = rep
                elif len(dates) == C.REP_K6_1_N:
                    out['rep_k6_1'].setdefault(s, {})[m] = beyond
        out['ok'] = not (out['missing'] or out['rep_months'] or out['units_bad'] or out['member_issues'] or out['held'])
        return out

    def _valuecheck(self, y):
        vc = os.path.join(C.CODE, 'c7a_valuecheck.py')
        p = subprocess.run([sys.executable, '-I', '-B', vc, '--year', y, '--out-root', self.out_root, '--h-input', self.h_input], capture_output=True, timeout=6 * 3600, creationflags=NOWIN)
        res = C.read_json(os.path.join(self.ydir(y), 'VALUECHECK_%s.json' % y), {})
        return (p.returncode, res)

    def finish_year(self, y, pins, passed, st):
        ks = self.k_checks(y, pins, passed)
        ok = all((ks[k]['ok'] for k in ('K1', 'K2', 'K3', 'K4', 'K5', 'K6')))
        C.write_json(os.path.join(self.ydir(y), 'K_CHECKS_%s.json' % y), ks)
        C.write_json(os.path.join(self.run_dir, 'K_CHECKS_%s.json' % y), ks)
        line = 'C-7a (4) %s: %s; K1 %s, K2 %s (flags %d), K3 %s, K4 %s, K5 %s, K6 %s; %d extracts; %s' % (y, 'PASS' if ok else 'FAIL', *['PASS' if ks[k]['ok'] else 'FAIL' for k in ('K1', 'K2')], len(ks['K2']['flags']), *['PASS' if ks[k]['ok'] else 'FAIL' for k in ('K3', 'K4', 'K5', 'K6')], len(self.manifest(y)), C.stamp(self.clock()))
        self.log(line)
        if not ok:
            st['year_status'] = 'FAILED (4): a manual decision is needed (one re-save); C-7a is not ended by this run'
            self.alert('C-7a (4)', line)
            return
        lp = os.path.join(self.ydir(y), 'C7A_4_LINE_%s_ASCII.txt' % y)
        with open(lp, 'w', encoding='ascii', newline='\n') as fh:
            fh.write('\n'.join(C.wrap(line)) + '\n')
        st['line4'] = line
        rc, vres = self.valuecheck(y)
        st['value_check'] = {'rc': rc, 'verdict': vres.get('verdict'), 'n_members': vres.get('n_members'), 'n_elements_differing': vres.get('n_elements_differing')}
        self.log('VALUE CHECK %s: %s (exit %s)' % (y, vres.get('verdict'), rc))
        if rc != 0 or vres.get('verdict') != 'PASS':
            st['year_status'] = 'HELD: the value check did not pass'
            self.alert('VALUE CHECK', '%s: %s; the extracts are held' % (y, vres.get('verdict')))
            return
        pub = self.publish(y)
        st['published'] = pub
        st['year_status'] = 'PUBLISHED' if pub.get('ok') else 'HELD: the copy to xfer\\out failed'
        self.log('PUBLISH %s: %s' % (y, json.dumps(pub)[:300]))

    def publish(self, y):
        src, dst = (self.ydir(y), os.path.join(self.xfer_out, y))
        man = self.manifest(y)
        files = sorted((r['path'] for r in man.values()))
        files += ['MANIFEST_%s.csv' % y, 'K_CHECKS_%s.json' % y, 'VALUECHECK_%s.json' % y, 'VALUECHECK_%s_ASCII.txt' % y, 'STATE_%s.json' % y]
        sums, bad = ([], [])
        for rel in files + ['C7A_4_LINE_%s_ASCII.txt' % y]:
            if rel.startswith('C7A_4_LINE_') and bad:
                bad.append('the (4) line not copied: a copy above failed')
                break
            a, b = (os.path.join(src, rel), os.path.join(dst, rel))
            if not os.path.exists(a):
                bad.append('absent: %s' % rel)
                continue
            ha, n = C.sha256_file(a)
            if os.path.exists(b):
                if C.sha256_file(b)[0] != ha:
                    bad.append('differs at the destination: %s' % rel)
                    continue
            else:
                os.makedirs(os.path.dirname(b), exist_ok=True)
                shutil.copyfile(a, b + '.partial')
                os.replace(b + '.partial', b)
            if C.sha256_file(b)[0] != ha:
                bad.append('copy hash: %s' % rel)
            sums.append('%s  %d  %s' % (ha, n, rel.replace('\\', '/')))
            if rel == files[-1]:
                sp = os.path.join(dst, 'SHA256SUMS_%s.txt' % y)
                with open(sp, 'w', encoding='ascii', newline='\n') as fh:
                    fh.write('\n'.join(sums) + '\n')
        return {'ok': not bad, 'files': len(sums), 'bad': bad[:10]}

    def cf_check(self):
        now = self.clock()
        open_years = [y for y in self.years if not str(self.state(y).get('year_status') or '').startswith('PUBLISHED')]
        ap = os.path.join(self.run_dir, 'ALERTS_ASCII.txt')
        if not self.cf_fired and os.path.exists(ap) and ('] TRIGGER CF:' in C.read_bytes(ap).decode('ascii', 'replace')):
            self.cf_fired = True
        if self.cf_fired or not open_years:
            return
        why = None
        if now >= C.parse_stamp(C.CF_DEADLINE):
            why = 'Sat 24 Oct 12:00 passed with %s not published' % ','.join(open_years)
        else:
            proj = self.projection(open_years)
            if proj is not None and proj > C.parse_stamp(C.CF_DEADLINE):
                why = 'projected end %s passes Sat 24 Oct 12:00' % C.stamp(proj)
        if why:
            self.cf_fired = True
            self.alert('TRIGGER CF', '%s; the user weighs route C under the gates; this run never starts it' % why)

    def projection(self, open_years):
        now = self.clock()
        pend_dl, pend_ex, recent, total = (0, 0, 0, 0)
        for y in open_years:
            pins, _ = pins_of(y, self.pins_fmt, self.check_pins)
            passed = k4_pass(y, pins, self.h_input, self.k4_tables)
            st = self.state(y)
            for n, r in pins.items():
                if not n.endswith('.tar'):
                    continue
                if n not in passed:
                    pend_dl += int(r['bytes'])
                if st['tars'].get(n, {}).get('status') != 'done':
                    pend_ex += int(r['bytes'])
            for n, r in passed.items():
                try:
                    t = C.parse_stamp(r.get('hashed_at', ''))
                except ValueError:
                    continue
                if (now - t).total_seconds() <= 6 * 3600:
                    recent += int(r['bytes'])
        r_dl = recent / (6 * 3600.0)
        secs = [s for k, b, s in self.timings[-5:]]
        byts = [b for k, b, s in self.timings[-5:]]
        r_ex = sum(byts) / sum(secs) if secs and sum(secs) > 0 else None
        tail = 2 * statistics.median(secs) + 1800 if secs else 1800
        cands = []
        if pend_dl and r_dl > 0:
            cands.append(pend_dl / r_dl + tail)
        if pend_ex and r_ex:
            cands.append(pend_ex / r_ex + 1800)
        if not cands:
            return None
        proj = now + datetime.timedelta(seconds=max(cands))
        ap = os.path.join(self.run_dir, 'ALERTS_ASCII.txt')
        if not self.t3_alerted and os.path.exists(ap) and ("] T-3: the downloads'" in C.read_bytes(ap).decode('ascii', 'replace')):
            self.t3_alerted = True
        if not self.t3_alerted and pend_dl and (r_dl > 0) and (now + datetime.timedelta(seconds=pend_dl / r_dl) > C.parse_stamp(C.T3_DEADLINE)):
            self.t3_alerted = True
            self.alert('T-3', "the downloads' own pace (%.1f MB/s over 6 h, from the K4 rows) projects the last pinned tar past Thu 22 Oct 24:00; an alert is written" % (r_dl / 1000000.0))
        return proj

    def run_once(self):
        worked = False
        all_done = True
        for y in self.years:
            st = self.state(y)
            if st.get('year_status'):
                continue
            all_done = False
            pins, _ = pins_of(y, self.pins_fmt, self.check_pins)
            passed = k4_pass(y, pins, self.h_input, self.k4_tables)
            tars = sorted((n for n in pins if n.endswith('.tar')))
            tars.sort(key=lambda n: (C.kind_of_tar(n) != 'stackgroups', n))
            sg_done = any((st['tars'].get(t, {}).get('status') == 'done' for t in tars if C.kind_of_tar(t) == 'stackgroups'))
            for t in tars:
                if st['tars'].get(t, {}).get('status') in ('done', 'failed'):
                    continue
                cn = t[:-4] + '.contents.txt'
                if t not in passed or cn not in passed:
                    continue
                if C.kind_of_tar(t) == 'ptsectors' and (not sg_done):
                    continue
                g = self.gates()
                if g != 'go':
                    self.save_state(y, st)
                    return g
                info = self.process_tar(y, t, pins, st)
                self.save_state(y, st)
                self.log('TAR %s %s: %s, %s members, %s s' % (y, t, info['status'], info.get('members_done'), info.get('seconds')))
                worked = True
                if C.kind_of_tar(t) == 'stackgroups' and info['status'] == 'done':
                    sg_done = True
                self.cf_check()
            stat = [st['tars'].get(t, {}).get('status') for t in tars]
            if len(tars) == 25 and all((s == 'done' for s in stat)):
                g = self.gates()
                if g != 'go':
                    return g
                self.finish_year(y, pins, passed, st)
                self.save_state(y, st)
                worked = True
            elif 'failed' in stat and all((s in ('done', 'failed') for s in stat)):
                st['year_status'] = 'FAILED (K2 or K3): a manual decision is needed (one re-save); C-7a is not ended by this run'
                self.alert('C-7a (4)', '%s: %s' % (y, st['year_status']))
                self.save_state(y, st)
        if all_done:
            return 'done'
        return 'worked' if worked else 'idle'

    def run(self, once=False):
        self.once = once
        base = t2_baseline(self.start, now=self.clock())
        self.log("RUN %s: years %s, P-1 %s, floor %.1f + peak %.1f GB, T-2 since %s (the later of the run's start %s and the note %s), code %s" % ('PASS' if once else 'START', ','.join(self.years), 'on' if self.p1 else 'off', C.MEM_FLOOR_GB, C.PEAK_GB, C.stamp(base) if base else 'none (no usable note)', C.stamp(self.start), note_name(T2_BASE_NOTE), json.dumps(self.code_sha16, sort_keys=True)))
        g = self.gate_t2(force=True)
        if g != 'go':
            self.stop_reason = g
            return g
        while True:
            r = self.run_once()
            self.write_status(r)
            if r == 'done':
                self.log('RUN END: every year published, held or failed')
                return r
            if r not in ('idle', 'worked'):
                self.log('RUN STOP: %s' % r)
                self.stop_reason = r
                return r
            if self.clock() >= C.parse_stamp(C.C7A_DEADLINE):
                self.alert('C-7a (1)', 'Sun 25 Oct 24:00 passed; the run ends; the years not published are listed')
                return 'C-7a (1) deadline'
            self.cf_check()
            if once and r == 'idle':
                return 'idle'
            if r == 'idle':
                for _ in range(max(1, C.POLL_S // 10)):
                    if os.path.exists(C.STOP_FILE):
                        break
                    self.sleep(min(10, C.POLL_S))

    def write_status(self, r):
        lines = ['C-7a EXTRACTION STATUS %s (%s)' % (C.stamp(self.clock()), r)]
        for y in self.years:
            st = self.state(y)
            done = sum((1 for v in st['tars'].values() if v.get('status') == 'done'))
            lines.append('%s: %d of 25 tars done; %d extracts; held %s; %s' % (y, done, len(self.manifest(y)), sorted(st['held']) or 'none', st.get('year_status') or 'open'))
        os.makedirs(self.run_dir, exist_ok=True)
        p = os.path.join(self.run_dir, 'STATUS_ASCII.txt')
        out = []
        for ln in lines:
            out.extend(C.wrap(C.mask(ln)))
        C.check_lines(out)
        with open(p + '.partial', 'w', encoding='ascii', newline='\n') as f:
            f.write('\n'.join(out) + '\n')
        os.replace(p + '.partial', p)
CROSS = (('merged', None, '20170701'), ('pt', 'ptegu', '20190701'), ('pt', 'ptnonipm', '20190709'))

def crossroute(cross_dir=C.CROSS_DIR, laptop_dir=C.XFER_IN_CROSS, run_dir=C.RUN_DIR, p1=False):
    fp = C.read_json(os.path.join(cross_dir, 'FETCH_RESULT.json'), {})
    out = {'started': C.stamp(), 'items': {}, 'verdict': 'FAIL'}
    if not fp.get('ok'):
        out['why'] = 'no FETCH_RESULT.json with ok true'
        C.write_json(os.path.join(run_dir, 'CROSSROUTE_RESULT.json'), out)
        return 1
    sums = {}
    sp_file = os.path.join(laptop_dir, 'SHA256SUMS.txt')
    if os.path.exists(sp_file):
        text = C.read_bytes(sp_file).decode('ascii', 'replace')
        for ln in text.splitlines():
            p = ln.split()
            if len(p) >= 2:
                sums[p[-1].replace('\\', '/').split('/')[-1]] = p[0]
    allok = True
    for kind, sec, d in CROSS:
        it = fp['members']['%s_%s_%s' % (kind, sec or '', d)]
        mpath = os.path.join(cross_dir, 'aws', it['file'])
        m = {'name': it['member'], 'data_offset': 0, 'size': os.path.getsize(mpath)}
        rec = {'member_sha256_fetch': it['sha256']}
        with open(mpath, 'rb') as fh:
            sel = R.read_selective(fh, m, 'nc4', 'merged' if kind == 'merged' else 'stack')
            who = R.read_whole(fh, m, 'nc4', 'merged' if kind == 'merged' else 'stack')
        rec['member_sha256_whole'] = who['member_sha256']
        rec['fetch_hash_equal'] = who['member_sha256'] == it['sha256']
        if kind == 'merged':
            mine = {sp: sel['means'][sp] for sp in C.MERGED_SPECIES}
            mine_w = {sp: who['means'][sp] for sp in C.MERGED_SPECIES}
            lname = 'emis_merged_%s.npz' % d
        else:
            gp = os.path.join(cross_dir, 'aws', fp['groups'][sec]['file'])
            gm = {'name': fp['groups'][sec]['member'], 'data_offset': 0, 'size': os.path.getsize(gp)}
            with open(gp, 'rb') as gh:
                g = R.read_groups(gh, gm, 'nc4')
            mine, info = R.cell_sums(sel, g, C.BASE_EXPECTED, False)
            mine_w, _ = R.cell_sums(who, g, C.BASE_EXPECTED, False)
            rec.update(n_stacks=info['n_stacks'], n_summed=info['n_summed'], n_outside=info['n_outside'])
            lname = 'pt_%s_%s.npz' % (sec, d)
        lp = os.path.join(laptop_dir, lname)
        if not os.path.exists(lp):
            rec['local'] = 'NOT FOUND'
            allok = False
            out['items'][lname] = rec
            continue
        lsha = C.sha256_file(lp)[0]
        rec['laptop_sha256'] = lsha
        rec['laptop_list_equal'] = sums.get(lname) == lsha
        with np.load(lp, allow_pickle=False) as zf:
            z = {k: zf[k] for k in zf.files}
        diffs = {}
        for k, a in mine.items():
            b = z[k] if k in z else None
            if b is None or b.shape != a.shape or b.dtype != a.dtype:
                diffs[k] = 'shape or dtype'
                continue
            diffs[k] = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
            diffs[k + '_whole_route'] = int(np.count_nonzero(mine_w[k].view(np.uint32) != b.view(np.uint32)))
        rec['elements_differing'] = diffs
        meta_ok = str(z['date']) == d and int(z['tflag_date']) == C.yyyyddd(d)
        if kind == 'pt':
            meta_ok = meta_ok and int(z['n_stacks']) == rec['n_stacks'] and (int(z['n_summed']) == rec['n_summed']) and (int(z['n_outside']) == rec['n_outside']) and (int(z['base']) == C.BASE_EXPECTED)
        rec['metadata_equal'] = bool(meta_ok)
        ok = rec['fetch_hash_equal'] and meta_ok and all((v == 0 for v in diffs.values())) and (rec['laptop_list_equal'] is not False)
        rec['ok'] = bool(ok)
        allok = allok and ok
        out['items'][lname] = rec
    out['verdict'] = 'PASS' if allok else 'FAIL'
    out['ended'] = C.stamp()
    out['code_sha16'] = Run([], False).code_sha16
    C.write_json(os.path.join(run_dir, 'CROSSROUTE_RESULT.json'), out)
    C.write_json(os.path.join(cross_dir, 'CROSSROUTE_RESULT.json'), out)
    return 0 if allok else 1

def go_line(path=None):
    path = path or os.path.join(C.PKG, 'GO_ASCII.txt')
    if not os.path.exists(path):
        return None
    text = C.read_bytes(path).decode('ascii', 'replace')
    out = {}
    for ln in text.split('\n'):
        s = ln.strip()
        if s.startswith('FREEZE: GO'):
            out['freeze'] = s
        if s.startswith('P-1:') and s.split(':', 1)[1].strip() in ('on', 'off'):
            out['p1'] = s.split(':', 1)[1].strip()
    return out if 'freeze' in out and 'p1' in out else None

def selftest_pins():
    import hashlib
    bad = []
    lines = C.read_bytes(os.path.join(C.VENDOR, 'PINS_ASCII.txt')).decode('ascii').splitlines()
    n = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if len(s) == 64 and all((c in '0123456789abcdef' for c in s)):
            rel = lines[i - 1].strip()
            h = hashlib.sha256(C.read_bytes(os.path.join(C.VENDOR, *rel.split('\\')))).hexdigest()
            n += 1
            if h != s:
                bad.append('vendor %s differs' % rel)
    if n != 6:
        bad.append('%d pins read, not 6' % n)
    for path, want in ((C.T2_SCRIPT, C.T2_SHA16), (C.SLOT_SCRIPT, C.SLOT_SHA16)):
        if C.sha16(path) != want:
            bad.append('%s changed' % os.path.basename(path))
    vers = {}
    import h5py
    import psutil
    import pyproj
    vers = {'python': sys.version.split()[0], 'numpy': np.__version__, 'h5py': h5py.__version__, 'hdf5': h5py.version.hdf5_version, 'pyproj': pyproj.__version__, 'psutil': psutil.__version__}
    want = {'numpy': '2.4.2', 'h5py': '3.15.1', 'hdf5': '1.14.6', 'pyproj': '3.7.2', 'psutil': '7.2.2'}
    for k, v in want.items():
        if vers[k] != v:
            bad.append('%s %s, not %s' % (k, vers[k], v))
    if not vers['python'].startswith('3.13.'):
        bad.append('python %s' % vers['python'])
    return {'ok': not bad, 'bad': bad, 'versions': vers, 'pins_read': n}

def reset_tar(y, tar, out_root=C.OUT_ROOT, run_dir=C.RUN_DIR):
    p = os.path.join(out_root, y, 'STATE_%s.json' % y)
    st = C.read_json(p)
    if st is None or tar not in st.get('tars', {}):
        print('NOT RESET: no state or no entry for %s' % tar)
        return 1
    k = 1
    while os.path.exists(p[:-5] + '_Backup_PriorToReset%d.json' % k):
        k += 1
    shutil.copyfile(p, p[:-5] + '_Backup_PriorToReset%d.json' % k)
    del st['tars'][tar]
    st['issues'] = [i for i in st['issues'] if tar not in i]
    st['year_status'] = None
    for key in ('line4', 'value_check', 'published'):
        st.pop(key, None)
    C.write_json(p, st)
    C.write_json(os.path.join(run_dir, 'STATE_%s.json' % y), st)
    C.append_record(os.path.join(run_dir, 'RUN_LOG_ASCII.txt'), '%s RESET %s %s (backup %d)' % (C.stamp(), y, tar, k))
    print('RESET %s %s' % (y, tar))
    return 0

def recheck_year(run, y, p1, go, selftest, cross):
    if go is None or go.get('p1') != p1:
        print("NOT REVERIFIED: GO_ASCII.txt lacks 'FREEZE: GO' or names P-1 %s, not %s" % (go and go.get('p1'), p1))
        return 5
    if not selftest.get('ok'):
        print('NOT REVERIFIED: SELFTEST FAILED: %s' % selftest.get('bad'))
        return 2
    if cross.get('verdict') != 'PASS' or cross.get('code_sha16') != run.code_sha16:
        print('NOT REVERIFIED: the cross-route check has not passed with this code')
        return 3
    need = C.MEM_FLOOR_GB + C.PEAK_GB
    if run.mem() < need:
        print('NOT REVERIFIED: memory %.2f GB, under %.1f GB' % (run.mem(), need))
        return 11
    names = ('STATE_%s.json' % y, 'K_CHECKS_%s.json' % y)
    srcs = [os.path.join(d, n) for d in (run.ydir(y), run.run_dir) for n in names]
    why, pins = (None, {})
    try:
        st = C.read_json(srcs[0])
    except ValueError:
        st, why = (None, 'the state is not readable')
    if why is None and st is None:
        why = 'no state %s' % note_name(srcs[0])
    elif why is None and st.get('p1') != p1:
        why = "the state's P-1 is %s, not %s" % (st.get('p1'), p1)
    if why is None:
        try:
            pins, _ = pins_of(y, run.pins_fmt, run.check_pins)
        except (OSError, KeyError, RuntimeError, ValueError) as e:
            why = 'pins_%s.csv not read (%s)' % (y, type(e).__name__)
    if why is None:
        tars = sorted((n for n in pins if n.endswith('.tar')))
        stat = [st.get('tars', {}).get(t, {}).get('status') for t in tars]
        vs = str(st.get('year_status') or '')
        absent = [note_name(s) for s in srcs if not os.path.exists(s) or os.path.getsize(s) == 0]
        if len(tars) != 25 or not all((s == 'done' for s in stat)):
            why = '%d of the %d pinned tars done, not 25 of 25' % (stat.count('done'), len(tars))
        elif not vs.startswith(('FAILED (4)', 'HELD')):
            why = 'the verdict is %s, not FAILED (4) or HELD' % (vs[:40] or 'open')
        elif absent:
            why = 'absent or empty: %s' % ', '.join(absent)
    if why is not None:
        print('NOT REVERIFIED: %s: %s' % (y, why))
        return 6
    g = run.gate_t2(force=True)
    if g != 'go':
        print('NOT REVERIFIED: %s: %s (run\\ALERTS_ASCII.txt)' % (y, g))
        return 20
    k = 1
    while any((os.path.exists(s[:-5] + '_Backup_PriorToRecheck%d.json' % k) for s in srcs)):
        k += 1
    for s in srcs:
        b = s[:-5] + '_Backup_PriorToRecheck%d.json' % k
        with open(s, 'rb') as fi, open(b, 'xb') as fo:
            shutil.copyfileobj(fi, fo)
        if os.path.getsize(b) == 0 or C.sha256_file(b)[0] != C.sha256_file(s)[0]:
            print('NOT REVERIFIED: %s: the backup %s differs from its source' % (y, note_name(b)))
            return 6
    st['year_status'] = None
    for key in ('line4', 'value_check', 'published'):
        st.pop(key, None)
    run.save_state(y, st)
    base = t2_baseline(run.start, now=run.clock())
    run.log('REVERIFY %s: P-1 %s, T-2 since %s; backups %s and %s (_Backup_PriorToRecheck%d) in %s and %s; year_status, line4, value_check and published cleared; code %s' % (y, p1, C.stamp(base) if base else 'none', names[0], names[1], k, note_name(run.ydir(y)), note_name(run.run_dir), json.dumps(run.code_sha16, sort_keys=True)))
    passed = k4_pass(y, pins, run.h_input, run.k4_tables)
    run.finish_year(y, pins, passed, st)
    run.save_state(y, st)
    run.write_status('reverify-year %s' % y)
    vs = str(st.get('year_status') or '')
    print('REVERIFY %s: %s' % (y, vs))
    if vs == 'PUBLISHED':
        return 0
    return 4 if vs.startswith('HELD') else 1

def plan_year(y, h_input=C.H_INPUT, run_dir=C.RUN_DIR):
    pins, s16 = pins_of(y)
    passed = k4_pass(y, pins, h_input, C.K4_TABLES)
    plan = C.read_json(C.S2A_PLAN, {})
    out = {'year': y, 'pins_sha16': s16, 'k4_pass': len(passed), 'tars': {}}
    for n in sorted(pins):
        if not n.endswith('.tar') or n not in passed or n[:-4] + '.contents.txt' not in passed:
            continue
        path = os.path.join(h_input, y, 'emis', n)
        rec = {'k2': T.k2(path, pins[n]['bytes'], ref_sizes(n, plan))}
        ms, w = T.walk(path)
        rec['k3'] = T.k3(ms, C.read_bytes(path[:-4] + '.contents.txt'))
        need, issues = T.pick(ms, y)
        by = {}
        for k, s, d in need:
            by.setdefault('%s %s' % (k, s or ''), []).append(d or '')
        rec['needed'] = {k: [len(v), min(v), max(v)] for k, v in by.items()}
        rec['issues'] = issues[:20]
        out['tars'][n] = rec
    C.write_json(os.path.join(run_dir, 'PLAN_%s.json' % y), out)
    return out

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('selftest-pins', 'plan', 'crossroute', 'run', 'status', 'reset-tar', 'reverify-year'))
    ap.add_argument('--year')
    ap.add_argument('--tar')
    ap.add_argument('--years', default='2015,2016,2018')
    ap.add_argument('--p1', choices=('on', 'off'))
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args(argv)
    if a.command == 'selftest-pins':
        r = selftest_pins()
        print(json.dumps(r, sort_keys=True))
        return 0 if r['ok'] else 1
    if a.command == 'plan':
        r = plan_year(a.year)
        print('PLAN %s: %d tars listed; see run\\PLAN_%s.json' % (a.year, len(r['tars']), a.year))
        return 0
    if a.command == 'crossroute':
        if go_line() is None:
            print("NO GO: GO_ASCII.txt lacks 'FREEZE: GO'")
            return 5
        st = selftest_pins()
        if not st['ok']:
            print('SELFTEST FAILED: %s' % st['bad'])
            return 2
        need = C.MEM_FLOOR_GB + C.PEAK_GB
        if C.mem_free_gb() < need:
            print('MEMORY %.2f GB, under %.1f GB: not run' % (C.mem_free_gb(), need))
            return 11
        rc = crossroute()
        print('CROSSROUTE %s' % ('PASS' if rc == 0 else 'FAIL'))
        return rc
    if a.command == 'reset-tar':
        return reset_tar(a.year, a.tar)
    if a.command == 'reverify-year':
        if a.year is None or a.p1 is None:
            ap.error('reverify-year needs --year Y and --p1 on or off')
        r = Run(C.YEARS, a.p1 == 'on')
        return recheck_year(r, a.year, a.p1, go_line(), selftest_pins(), C.read_json(os.path.join(C.RUN_DIR, 'CROSSROUTE_RESULT.json'), {}))
    if a.command == 'status':
        for y in a.years.split(','):
            p = os.path.join(C.RUN_DIR, 'STATE_%s.json' % y)
            print(y, C.read_bytes(p).decode('ascii')[:2000] if os.path.exists(p) else 'no state')
        return 0
    if a.p1 is None:
        ap.error('run needs --p1 on or off (as the design fixes it)')
    st = selftest_pins()
    if not st['ok']:
        print('SELFTEST FAILED: %s' % st['bad'])
        return 2
    go = go_line()
    if go is None or go.get('p1') != a.p1:
        print("NO GO: GO_ASCII.txt lacks 'FREEZE: GO' or names P-1 %s, not %s" % (go and go.get('p1'), a.p1))
        return 5
    cr = C.read_json(os.path.join(C.RUN_DIR, 'CROSSROUTE_RESULT.json'), {})
    if cr.get('verdict') != 'PASS' or cr.get('code_sha16') != Run([], False).code_sha16:
        print('the cross-route check has not passed with this code; the run waits')
        return 3
    r = Run(a.years.split(','), a.p1 == 'on')
    res = r.run(once=a.once)
    print('RUN %s' % res)
    return {'done': 0, 'idle': 10, 'worked': 10, 'memory': 11}.get(res, 20)
if __name__ == '__main__':
    sys.exit(main())
