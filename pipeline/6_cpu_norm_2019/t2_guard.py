"""Copies of the spend guard's functions, which read each runtime's compute-unit rate and stop the stage when its recorded spend would pass the limit."""
from tier_a_rules import AUTHORS_FORM, _num, _rate_field, typed_for_runtime

def max_rate(c):
    vals = [g.get('units_per_hour') for g in c.get('gpu_candidates') or [] if isinstance(g, dict)] + [c.get('cpu_units_per_hour')]
    nums = [float(v) for v in vals if _num(v) and v > 0]
    return max(nums) if nums else None

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
