"""The roles guard: gives each day of a fold its role (training, validation, test, held-back or buffer) from the splits and the held-back days, and refuses any read of a day for a purpose that its role does not allow."""
ROLES = ('TRAIN', 'VAL', 'TEST_CV', 'LOCKBOX', 'BUFFER')
ALLOWED = {'stat': ('TRAIN',), 'fit': ('TRAIN',), 'threshold': ('TRAIN',), 'peak': ('TRAIN',), 'validate': ('VAL',), 'early_stop': ('VAL',), 'predict_cv': ('TEST_CV',), 'predict_lockbox': ('LOCKBOX',)}
PARTS = (('train', 'TRAIN'), ('validation', 'VAL'), ('held_out', 'TEST_CV'))

class GuardError(AssertionError):
    pass

def _int_list(v, what):
    if not isinstance(v, (list, tuple)) or not all((isinstance(i, int) and (not isinstance(i, bool)) for i in v)):
        raise ValueError('%s: not a list of day indices' % what)
    return [int(i) for i in v]

def year_map(doc, what):
    if not isinstance(doc, dict):
        raise ValueError('%s is not year -> day indices' % what)
    out = {}
    for k, v in doc.items():
        if not (isinstance(k, str) and len(k) == 4 and k.isdigit()):
            raise ValueError('%s: the key %r is not a year' % (what, k))
        out[int(k)] = set(_int_list(v, '%s %s' % (what, k)))
    return out

def part_days(part, what='a split part'):
    if isinstance(part, dict):
        if 'index_by_year' not in part:
            raise ValueError('%s has no index_by_year' % what)
        m = year_map(part['index_by_year'], what)
        return sorted(((y, i) for y, s in m.items() for i in s))
    out = []
    for d in part:
        out.append((int(d[0]), int(d[1])))
    if len(set(out)) != len(out):
        raise ValueError('%s lists a day twice' % what)
    return sorted(out)

def lockbox_maps(lb_doc):
    if not isinstance(lb_doc, dict) or 'days' not in lb_doc or 'lockbox' not in lb_doc:
        raise ValueError('LOCKBOX_DAYS.json is not {days, lockbox}')
    days = year_map(lb_doc['days'], 'LOCKBOX_DAYS.json days')
    lock = year_map(lb_doc['lockbox'], 'LOCKBOX_DAYS.json lockbox')
    for y, s in lock.items():
        if not s <= days.get(y, set()):
            raise ValueError('LOCKBOX_DAYS.json %d: lockbox days outside "days"' % y)
    return (days, lock)

def roles_of(split, lb_doc=None):
    if not isinstance(split, dict):
        raise ValueError('the split is not a SPLITS.json entry')
    roles = {}
    held_years = set()
    for key, role in PARTS:
        if key not in split:
            raise ValueError('the split has no %r part' % key)
        for y, i in part_days(split[key], key):
            if (y, i) in roles:
                raise ValueError('day %d/%d is in two parts of the split' % (y, i))
            roles[y, i] = role
            if role == 'TEST_CV':
                held_years.add(y)
    if lb_doc is None:
        return roles
    days, lock = lockbox_maps(lb_doc)
    for y, s in days.items():
        for i in s:
            if i in lock.get(y, ()) and y in held_years:
                roles[y, i] = 'LOCKBOX'
            else:
                roles[y, i] = 'BUFFER'
    return roles

def allowed(purpose):
    if purpose not in ALLOWED:
        raise GuardError('unknown purpose %r (the purposes: %s)' % (purpose, ', '.join(sorted(ALLOWED))))
    return ALLOWED[purpose]

def roles_guard(days, purpose, roles):
    ok = allowed(purpose)
    n = 0
    for d in days:
        key = (int(d[0]), int(d[1]))
        role = roles.get(key)
        if role not in ok:
            raise GuardError('%s: day %d/%d has the role %s; %s takes %s only' % (purpose, key[0], key[1], role or 'none (in no part)', purpose, ' or '.join(ok)))
        n += 1
    return n

def days_with(roles, role, year=None):
    if role not in ROLES:
        raise ValueError('unknown role %r' % (role,))
    return sorted((k for k, r in roles.items() if r == role and (year is None or k[0] == int(year))))

def counts(roles):
    out = {r: 0 for r in ROLES}
    for r in roles.values():
        out[r] += 1
    return out
