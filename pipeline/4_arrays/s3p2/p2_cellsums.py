"""The reducer that sums per-stack emissions into grid cells, copied byte for byte from the Drive extraction code, so that both stages share one definition of the point and marine NOx channel."""
import sys
sys.path.insert(0, '.\\run\\code')
import numpy as np
from choi import stacks

class C:
    GRID = (299, 459)
    PT_SPECIES = ('NO', 'NO2', 'HONO')

def cell_sums(res, g, base, p1):
    grids, info = ({}, None)
    for sp in C.PT_SPECIES:
        grid, info = stacks.sum_to_cells(res['means'][sp], g['row'], g['col'], C.GRID[0], C.GRID[1], base=base)
        grids[sp] = grid.astype(np.float32)
    if p1:
        tot = res['means']['NO'] + res['means']['NO2'] + res['means']['HONO']
        grid, _ = stacks.sum_to_cells(tot, g['row'], g['col'], C.GRID[0], C.GRID[1], base=base)
        grids['P1_NOX'] = grid.astype(np.float32)
    return (grids, info)
