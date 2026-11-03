"""Compute the structure hash of every module listed in
CODE_AS_RUN.txt (the sha256 of the module's syntax tree with every
docstring removed and every string constant emptied) and compare it with
the hash listed for the deposited file. The same hash of each file as
it ran is listed beside it, so that a copy of the file as it ran can be
tested the same way. Standard library only. Run from the
repository root: python pipeline/build_structure_hashes.py
The hashes were computed with Python 3.13; the syntax-tree dump, and so
the hash, differs between Python versions, so run the script with
Python 3.13."""
import ast
import hashlib
import os
import sys


def structure_hash(path):
    tree = ast.parse(open(path, "rb").read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                node.body = body[1:] or ([] if isinstance(node, ast.Module)
                                         else [ast.Pass()])
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()


def main():
    if sys.version_info[:2] != (3, 13):
        print("The hashes were computed with Python 3.13; the syntax-tree "
              "dump, and so the hash, differs between Python versions, so "
              "run the script with Python 3.13.")
    here = os.path.dirname(os.path.abspath(__file__))
    rows = [ln.split("\t") for ln in open(os.path.join(
        here, "CODE_AS_RUN.txt"), encoding="ascii").read().splitlines()
        if ln and not ln.startswith("#")]
    same = differ = 0
    for r in rows:
        path, deposited = r[0], r[3]
        got = structure_hash(os.path.join(here, *path.split("/")))
        ok = got == deposited
        same += ok
        differ += not ok
        print(("MATCH  " if ok else "DIFFERS"), path)
    print("modules %d, match %d, differ %d" % (len(rows), same, differ))
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
