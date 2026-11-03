"""Runs a trainer script as the main program after setting the overrides of the training package's constants that the configuration file lists; with an empty list the script runs unchanged."""
import ast
import importlib.abc
import importlib.machinery
import json
import os
import re
import runpy
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
DEC_ID = re.compile('^R-[0-9]{1,6}$')
USAGE = 'usage: train_shim.py --config TIER_A_CONFIG.json SCRIPT [ARGS ...]'

def load_overrides(path):
    with open(path, 'rb') as fh:
        conf = json.loads(fh.read().decode('ascii'))
    return overrides_of(conf)

def overrides_of(conf):
    ov = conf.get('trainer_overrides') or []
    if not isinstance(ov, list):
        raise SystemExit(4)
    table = {}
    for e in ov:
        ok = isinstance(e, dict) and isinstance(e.get('module'), str) and re.match('^train\\.[A-Za-z_][A-Za-z0-9_]*$', e['module']) and isinstance(e.get('name'), str) and e['name'].isidentifier() and ('value' in e) and DEC_ID.match(str(e.get('decided_by', '')))
        if not ok:
            print('SHIM REFUSED: a trainer_overrides entry is not {module train.<m>, name, value, decided_by R-<digits>}', flush=True)
            raise SystemExit(4)
        table.setdefault(e['module'], {})[e['name']] = e
    lb = conf.get('lockbox') or {}
    if lb.get('decided_by') is not None:
        if not (DEC_ID.match(str(lb.get('decided_by'))) and isinstance(lb.get('file'), str) and re.match('^[A-Za-z0-9_]+\\.json$', lb['file'])):
            print('SHIM REFUSED: the lockbox is not {decided_by R-<digits>, file <name>.json}', flush=True)
            raise SystemExit(4)
        if 'LOCKBOX_FILE' in table.get('train.data', {}):
            print("SHIM REFUSED: train.data LOCKBOX_FILE is set by the config's lockbox, not by trainer_overrides", flush=True)
            raise SystemExit(4)
        table.setdefault('train.data', {})['LOCKBOX_FILE'] = {'module': 'train.data', 'name': 'LOCKBOX_FILE', 'value': lb['file'], 'decided_by': lb['decided_by']}
    return table

def _literal(value, old):
    if isinstance(old, ast.Tuple) and isinstance(value, list):
        value = tuple(value)
    node = ast.parse(repr(value), mode='eval').body
    return ast.copy_location(node, old)

def rewrite(source, filename, names):
    tree = ast.parse(source, filename)
    done = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names:
                    node.value = _literal(names[tgt.id]['value'], node.value)
                    done.add(tgt.id)
                elif isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List)) and (len(tgt.elts) == len(node.value.elts)):
                    for k, el in enumerate(tgt.elts):
                        if isinstance(el, ast.Name) and el.id in names:
                            node.value.elts[k] = _literal(names[el.id]['value'], node.value.elts[k])
                            done.add(el.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and (node.target.id in names) and (node.value is not None):
            node.value = _literal(names[node.target.id]['value'], node.value)
            done.add(node.target.id)
    missing = sorted(set(names) - done)
    if missing:
        print('SHIM REFUSED: %s has no top-level assignment of %s' % (os.path.basename(filename), ', '.join(missing)), flush=True)
        raise SystemExit(4)
    ast.fix_missing_locations(tree)
    return compile(tree, filename, 'exec')

class _Loader(importlib.abc.Loader):

    def __init__(self, names):
        self.names = names

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = module.__spec__.origin
        with open(path, 'rb') as fh:
            src = fh.read().decode('utf-8')
        code = rewrite(src, path, self.names)
        print('SHIM %s: %s' % (module.__name__, '; '.join(('%s (%s)' % (n, e['decided_by']) for n, e in sorted(self.names.items())))), flush=True)
        exec(code, module.__dict__)

class _Finder(importlib.abc.MetaPathFinder):

    def __init__(self, table):
        self.table = table

    def find_spec(self, name, path, target=None):
        if name not in self.table:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or not str(spec.origin or '').endswith('.py'):
            print('SHIM REFUSED: %s is not a source module' % name, flush=True)
            raise SystemExit(4)
        spec.loader = _Loader(self.table[name])
        return spec

def main(argv):
    if len(argv) < 3 or argv[0] != '--config':
        raise SystemExit(USAGE)
    table = load_overrides(argv[1])
    already = sorted((m for m in table if m in sys.modules))
    if already:
        print('SHIM REFUSED: imported before the shim: %s' % already, flush=True)
        raise SystemExit(4)
    if table:
        sys.meta_path.insert(0, _Finder(table))
    script = os.path.abspath(argv[2])
    sys.argv = [script] + list(argv[3:])
    runpy.run_path(script, run_name='__main__')
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
