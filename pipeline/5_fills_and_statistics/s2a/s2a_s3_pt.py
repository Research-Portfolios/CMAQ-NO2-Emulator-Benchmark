"""Gives the U.S. federal holiday dates (legal and observed) used for the representative-day sectors, and holds the test of the point and marine route: the stack groups files, per-stack daily means read whole and selectively, the row and column check by the grid projection, and the stack-to-cell sums."""
import datetime as dt
import io
import os
import re
import sys
import tarfile
sys.path.insert(0, '/content/emul/run/code')
sys.path.insert(0, '/content/emul/run/code/s2a')
import numpy as np
from choi import stacks
from choi_dl import common, prov, s3io
import s2a_s3_common as C
SECTORS = ('ptegu', 'ptnonipm', 'pt_oilgas', 'othpt', 'cmv_c1c2_12', 'cmv_c3_12')
ORDER = ('ptegu', 'cmv_c3_12', 'cmv_c1c2_12', 'pt_oilgas', 'ptnonipm', 'othpt')
DAILY = ('ptegu', 'cmv_c1c2_12', 'cmv_c3_12')
SPECIES = ('NO', 'NO2', 'HONO')
CAP_S = 6 * 3600.0
R_EARTH = 6370000.0

def federal_holidays(y):

    def nth(m, wd, n):
        f = dt.date(y, m, 1)
        return f + dt.timedelta(days=(wd - f.weekday()) % 7 + 7 * (n - 1))

    def last(m, wd):
        d = dt.date(y, m + 1, 1) - dt.timedelta(days=1)
        return d - dt.timedelta(days=(d.weekday() - wd) % 7)
    legal = [("New Year's Day", dt.date(y, 1, 1)), ('Birthday of Martin Luther King, Jr.', nth(1, 0, 3)), ("Washington's Birthday", nth(2, 0, 3)), ('Memorial Day', last(5, 0)), ('Independence Day', dt.date(y, 7, 4)), ('Labor Day', nth(9, 0, 1)), ('Columbus Day', nth(10, 0, 2)), ('Veterans Day', dt.date(y, 11, 11)), ('Thanksgiving Day', nth(11, 3, 4)), ('Christmas Day', dt.date(y, 12, 25))]
    out = []
    for nm, d in legal:
        obs = d
        if d.weekday() == 5:
            obs = d - dt.timedelta(days=1)
        elif d.weekday() == 6:
            obs = d + dt.timedelta(days=1)
        out.append({'name': nm, 'legal': d.strftime('%Y%m%d'), 'observed': obs.strftime('%Y%m%d')})
    return out

def groups_member_name(names, sector, year):
    c = [n for n in names if re.search('/stack_groups_%s_12US1_' % sector, n)]
    nc4 = [n for n in c if n.endswith('.nc4')]
    return (nc4 or c or [None])[0]

def lcc_rowcol(lat, lon, a):
    import pyproj
    p = pyproj.Proj('+proj=lcc +lat_1=%r +lat_2=%r +lat_0=%r +lon_0=%r +x_0=0 +y_0=0 +a=%r +b=%r +units=m +no_defs' % (a['P_ALP'], a['P_BET'], a['YCENT'], a['P_GAM'], R_EARTH, R_EARTH))
    x, y = p(lon, lat)
    c0 = np.floor((x - a['XORIG']) / a['XCELL']).astype(np.int64)
    r0 = np.floor((y - a['YORIG']) / a['YCELL']).astype(np.int64)
    return (r0, c0, x, y)

def rowcol_check(g, a):
    r0, c0, x, y = lcc_rowcol(g['latitude'], g['longitude'], a)
    nr, nc = (int(a['NROWS']), int(a['NCOLS']))
    inside = (r0 >= 0) & (r0 < nr) & (c0 >= 0) & (c0 < nc)
    out = {'n_stacks': int(r0.size), 'n_outside_by_latlon': int((~inside).sum())}
    for base in (1, 0):
        ok = (g['row'] == r0 + base) & (g['col'] == c0 + base)
        out['agree_base%d' % base] = int((ok & inside).sum())
    base = 1 if out['agree_base1'] >= out['agree_base0'] else 0
    ok = (g['row'] == r0 + base) & (g['col'] == c0 + base)
    out['base'] = base
    out['agree'] = int((ok & inside).sum())
    out['disagree'] = int((~ok & inside).sum())
    out['file_rowcol_outside_grid'] = int(((g['row'] - base < 0) | (g['row'] - base >= nr) | (g['col'] - base < 0) | (g['col'] - base >= nc)).sum())
    bad = ~ok & inside
    if bad.any():
        dr = np.abs(g['row'][bad] - (r0[bad] + base))
        dc = np.abs(g['col'][bad] - (c0[bad] + base))
        out['disagree_max_cells'] = int(max(dr.max(), dc.max()))
        out['disagree_within_1_cell'] = int(((dr <= 1) & (dc <= 1)).sum())
        fx = (x[bad] - a['XORIG']) / a['XCELL'] % 1.0
        fy = (y[bad] - a['YORIG']) / a['YCELL'] % 1.0
        edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
        out['disagree_edge_distance_max_cell'] = float(edge.max())
    return out

def main():
    common.set_temp()
    t = C.timer()
    k = C.load_json(C.KEYS_JSON)
    heads = k['heads']
    mc = C.load_json(os.path.join(C.WORK_S3, 'mcip.json'))
    a = mc['projection']['attrs']
    client = s3io.S3Client(log=C.log)
    out = {'made': common.now_pair(), 'stackgroups': {}, 'groups_files': {}, 'tests': {}}
    sg_rates = []
    for y in C.YEARS:
        key = C.sg_key(y)
        h = heads[key]
        path = os.path.join(C.SG_DIR, key.split('/')[-1])
        if prov.verified(C.SG_DIR, path):
            info = {'skipped': True}
        else:
            info = client.stream_to_file(key, 0, h['size'], path, if_match=h['etag'])
            u, loc = common.now_pair()
            prov.append(C.SG_DIR, path=path, source='s3://%s/%s' % (s3io.BUCKET, key), byte_range='0-%d' % (h['size'] - 1), retrieved_utc=u, retrieved_local=loc, http_status=info.get('status'), bytes=os.stat(path).st_size, sha256=info['sha256'], version=C.VERSION_EMIS, etag=h['etag'], last_modified=h['last_modified'], note='stack groups tar fetched whole, kept raw (the point and marine route)')
            sg_rates.append(h['size'] / info['seconds'] / 1000000.0)
        with tarfile.open(path, 'r:') as tf:
            mem = [{'name': ti.name, 'size': int(ti.size), 'is_file': ti.isfile()} for ti in tf.getmembers()]
        out['stackgroups'][str(y)] = {'key': key, 'path': common.rel(path), 'transfer': info, 'members': mem}
        C.log('STACKGROUPS %d: %s; %d members' % (y, 'skipped (row re-verifies)' if info.get('skipped') else '%d bytes in %.1f s' % (info['bytes'], info['seconds']), len(mem)))
    gbytes = {}
    for y in C.YEARS:
        path = os.path.join(common.ROOT, out['stackgroups'][str(y)]['path'])
        names = [m['name'] for m in out['stackgroups'][str(y)]['members']]
        with tarfile.open(path, 'r:') as tf:
            for sec in SECTORS:
                gn = groups_member_name(names, sec, y)
                rec = {'member': gn}
                if gn is None:
                    rec['reason'] = 'no stack groups file for the sector'
                    out['groups_files']['%s_%d' % (sec, y)] = rec
                    continue
                b = tf.extractfile(tf.getmember(gn)).read()
                gbytes[sec, y] = b
                import netCDF4
                ds = netCDF4.Dataset('g.nc', mode='r', memory=b)
                try:
                    rec['data_model'] = ds.data_model
                    rec['variables'] = sorted(ds.variables)
                    rec['has'] = {v: v in ds.variables for v in ('ROW', 'COL', 'LATITUDE', 'LONGITUDE')}
                    g = stacks.groups_row_col(ds)
                    rec['n_stacks'] = g.get('n_stacks')
                    rec['reason'] = g.get('reason')
                    if not g.get('reason') and 'latitude' in g:
                        rec['rowcol_check'] = rowcol_check(g, a)
                finally:
                    ds.close()
                out['groups_files']['%s_%d' % (sec, y)] = rec
                C.log('GROUPS %s %d: %s, %s stacks, %s; ROW/COL %s' % (sec, y, gn.split('/')[-1], rec.get('n_stacks'), rec.get('data_model'), rec.get('rowcol_check')))
    key = C.pt_key(2019, 7)
    it = [x for x in k['groups']['ptsectors'] if x['key'] == key][0]
    tix = s3io.TarIndex(client, key, heads[key]['etag'], heads[key]['size'], it['members'])
    rates = []
    rp = s3io.ReaderProcess(client.limiter, timeout=1800)
    try:
        for sec, day in (('ptegu', '20190701'), ('ptnonipm', '20190709')):
            name = [m['name'] for m in it['members'] if '/inln_mole_%s_%s_' % (sec, day) in m['name']][0]
            m = tix.locate(name)
            whole, data = s3io.whole_read(client, key, m, heads[key]['etag'], SPECIES, 'stack')
            tr = whole['transfer']
            rates.append(tr['bytes'] / tr['seconds'] / 1000000.0)
            base = out['groups_files']['%s_2019' % sec].get('rowcol_check', {}).get('base', 1)
            st = stacks.stack_to_cell(data, gbytes[sec, 2019], species=SPECIES, base=base)
            import netCDF4
            ds = netCDF4.Dataset('m.nc', mode='r', memory=data)
            nc_model = ds.data_model
            nc_units = {sp: str(ds.variables[sp].units).strip() for sp in SPECIES if sp in ds.variables}
            ds.close()
            del data
            sel = rp.run({'key': key, 'member': m, 'etag': heads[key]['etag'], 'species': list(SPECIES), 'kind': 'stack', 'n_hours': 24, 'layer': 0})
            eq = {}
            if sel.get('ok'):
                for sp in SPECIES:
                    eq[sp] = bool(np.array_equal(whole['means'][sp], sel['means'][sp]))
            win = sel.get('window', {})
            ratio = win.get('bytes', 0) / float(m['size']) if sel.get('ok') else None
            g = None
            import netCDF4 as n4
            gds = n4.Dataset('g.nc', mode='r', memory=gbytes[sec, 2019])
            try:
                g = stacks.groups_row_col(gds)
            finally:
                gds.close()
            n_file = whole.get('n_stacks')
            counts_agree = n_file == g['n_stacks'] and (not sel.get('ok') or sel.get('n_stacks') == g['n_stacks'])
            sel_grids = {}
            sel_info = None
            if sel.get('ok') and counts_agree:
                for sp in SPECIES:
                    grd, sel_info = stacks.sum_to_cells(sel['means'][sp], g['row'], g['col'], base=base)
                    sel_grids[sp] = grd
            grids_equal = {sp: bool(np.array_equal(st['grids'][sp], sel_grids[sp])) for sp in SPECIES if sp in sel_grids and sp in st.get('grids', {})}
            passed = bool(sel.get('ok') and eq and all(eq.values()) and (len(eq) == len(SPECIES)) and (ratio <= 0.25))
            tag = '%s_%s' % (sec, day)
            for sp in SPECIES:
                np.save(os.path.join(C.WORK_S3, 'pt_%s_whole_stack_%s.npy' % (tag, sp)), whole['means'][sp])
                if sel.get('ok'):
                    np.save(os.path.join(C.WORK_S3, 'pt_%s_selective_stack_%s.npy' % (tag, sp)), sel['means'][sp])
                if sp in st.get('grids', {}):
                    np.save(os.path.join(C.WORK_S3, 'pt_%s_whole_cells_%s.npy' % (tag, sp)), st['grids'][sp])
                if sp in sel_grids:
                    np.save(os.path.join(C.WORK_S3, 'pt_%s_selective_cells_%s.npy' % (tag, sp)), sel_grids[sp])
            res = {'key': key, 'member': m, 'whole': {k2: v for k2, v in whole.items() if k2 != 'means'}, 'netcdf4': {'data_model': nc_model, 'units': nc_units}, 'selective': {k2: v for k2, v in sel.items() if k2 != 'means'}, 'equal_bitwise': eq, 'selective_ratio': ratio, 'passed': passed, 'rate_MB_per_s': rates[-1], 'n_stacks_file': n_file, 'n_stacks_groups': g['n_stacks'], 'counts_agree': counts_agree, 'base_used': base, 'stack_to_cell_whole': {k2: v for k2, v in st.items() if k2 not in ('grids', 'daily_means')}, 'selective_cells_info': sel_info, 'cell_grids_equal': grids_equal}
            out['tests'][tag] = res
            C.log('PT %s: whole %d bytes %.2f s (%.2f MB/s); selective ok %s, %d bytes in %d requests, %.2f s, ratio %s; equal %s; stacks file %s groups %s; cells equal %s; passed %s' % (tag, tr['bytes'], tr['seconds'], rates[-1], sel.get('ok'), win.get('bytes', 0), win.get('requests', 0), sel.get('seconds', 0.0), '%.4f' % ratio if ratio is not None else None, eq, n_file, g['n_stacks'], grids_equal, passed))
            if not sel.get('ok'):
                C.log('PT %s selective error: %s' % (tag, sel.get('error')))
    finally:
        rp.close()
    rate = float(np.mean(rates))
    both = all((out['tests'][t2]['passed'] for t2 in out['tests']))
    per = {}
    for itm in k['groups']['ptsectors']:
        yy = int(re.search('_(\\d{4})_(\\d{2})_EQUATES', itm['key']).group(1))
        mm = int(re.search('_(\\d{4})_(\\d{2})_EQUATES', itm['key']).group(2))
        for m in itm['members']:
            for sec in SECTORS:
                mt = re.search('/inln_mole_%s_(\\d{8})_12US1_' % sec, m['name'])
                if mt:
                    per.setdefault(sec, []).append({'year': yy, 'month': mm, 'date': mt.group(1), 'size': m['size'], 'name': m['name'], 'key': itm['key']})
    hol = {y: federal_holidays(y) for y in C.YEARS}
    hol_dates = {}
    for y in C.YEARS:
        for hh in hol[y]:
            hol_dates.setdefault(hh['legal'], []).append(hh['name'] + ' (legal date)')
            if hh['observed'] != hh['legal']:
                hol_dates.setdefault(hh['observed'], []).append(hh['name'] + ' (observed)')
    sec_tab = {}
    for sec in SECTORS:
        ms = per.get(sec, [])
        months = {}
        for x in ms:
            months.setdefault('%d-%02d' % (x['year'], x['month']), []).append(x['date'])
        on_hol = [{'date': x['date'], 'holiday': hol_dates[x['date']]} for x in ms if x['date'] in hol_dates]
        sec_tab[sec] = {'n_members': len(ms), 'bytes': int(sum((x['size'] for x in ms))), 'members_per_month': {mo: len(v) for mo, v in sorted(months.items())}, 'dates_per_month': {mo: sorted(v) for mo, v in sorted(months.items())}, 'members_on_federal_holidays': on_hol}
    t_eg = out['tests']['ptegu_20190701']
    t_nn = out['tests']['ptnonipm_20190709']
    proj = {'whole': {}, 'selective': {}}
    for sec in SECTORS:
        st_ = sec_tab[sec]
        proj['whole'][sec] = {'bytes': st_['bytes'], 'hours': (st_['bytes'] / (rate * 1000000.0) + 2 * st_['n_members']) / 3600.0}
        ref = t_eg if sec in DAILY else t_nn
        if ref['selective'].get('ok'):
            frac = ref['selective_ratio']
            nreq = 1 + ref['selective']['window']['requests']
            b = st_['bytes'] * frac
            proj['selective'][sec] = {'bytes': int(b), 'requests_per_member': nreq, 'reference': 'ptegu_20190701' if sec in DAILY else 'ptnonipm_20190709', 'hours': (b / (rate * 1000000.0) + nreq * st_['n_members']) / 3600.0}
    proj['whole_total_hours'] = sum((v['hours'] for v in proj['whole'].values()))
    proj['selective_total_hours'] = sum((v['hours'] for v in proj['selective'].values())) if proj['selective'] else None
    kept, dropped = ([], {})
    for sec in SECTORS:
        for y in C.YEARS:
            rec = out['groups_files'].get('%s_%d' % (sec, y), {})
            if rec.get('reason') or rec.get('member') is None:
                dropped[sec] = 'stack groups file of %d: %s' % (y, rec.get('reason') or 'missing')
    for tg, res in out['tests'].items():
        if not res['counts_agree']:
            dropped[tg.split('_2019')[0]] = 'stack counts disagree in the test member (file %s, groups %s)' % (res['n_stacks_file'], res['n_stacks_groups'])
    if both:
        route = 'selective'
        kept = [s for s in SECTORS if s not in dropped]
    else:
        route = 'whole'
        total = 0.0
        for sec in ORDER:
            if sec in dropped:
                continue
            h = proj['whole'][sec]['hours'] * 3600.0
            if total + h <= CAP_S:
                kept.append(sec)
                total += h
            else:
                dropped[sec] = 'its whole transfer for both years (%.2f h) would take this route over the 6 h cap' % (h / 3600.0)
    out.update({'rate_MB_per_s': rate, 'both_passed': both, 'route': route, 'sectors_kept': kept, 'sectors_dropped': dropped, 'projections': proj, 'sectors': sec_tab, 'federal_holidays': hol, 'seconds': t(), 'client': client.summary(), 'stackgroups_rates_MB_per_s': sg_rates})
    C.save_json(os.path.join(C.WORK_S3, 'pt_test.json'), out)
    for sec in SECTORS:
        C.log('SECTOR %s: %d members, %.2f GB; whole %.2f h; selective %s' % (sec, sec_tab[sec]['n_members'], sec_tab[sec]['bytes'] / 1000000000.0, proj['whole'][sec]['hours'], '%.2f h' % proj['selective'][sec]['hours'] if sec in proj['selective'] else None))
    C.log('PT ROUTE %s; kept %s; dropped %s; totals whole %.2f h, selective %s h' % (route, kept, dropped, proj['whole_total_hours'], None if proj['selective_total_hours'] is None else '%.2f' % proj['selective_total_hours']))
    C.log('PT done in %.1f s; requests %d' % (t(), client.stats.requests))
if __name__ == '__main__':
    main()
