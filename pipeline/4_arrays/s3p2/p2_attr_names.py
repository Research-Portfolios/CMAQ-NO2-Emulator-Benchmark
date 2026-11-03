"""Prints the names, and never the values, of the global attributes and variables of the MCIP files and the attribute names of chosen variables of one RSIG answer (read only)."""
import os
import sys
sys.path.insert(0, '.\\run\\code')
from s3 import s3common as S

def main():
    import netCDF4
    inputs = S.inventory_inputs()
    for e in sorted(inputs['mcip'], key=lambda x: x['path'])[:3]:
        p = S.path_abs(e['path'])
        ds = netCDF4.Dataset(p, 'r')
        try:
            print('== %s' % os.path.basename(p))
            print('  global attribute names: %s' % ' '.join(ds.ncattrs()))
            print('  variables (%d): %s' % (len(ds.variables), ' '.join(ds.variables)))
        finally:
            ds.close()
    idx = S.rsig_index(inputs)
    for ser, var in (('M4', 'UWIND'), ('M5', 'VWIND')):
        data, _chk = S.decompress_answer(idx[ser, '201501'])
        ds = netCDF4.Dataset('inmemory.nc', mode='r', memory=data)
        try:
            print('== RSIG %s 201501 %s' % (ser, var))
            print('  global attribute names: %s' % ' '.join(ds.ncattrs()))
            print('  %s attribute names: %s' % (var, ' '.join(ds.variables[var].ncattrs())))
        finally:
            ds.close()
if __name__ == '__main__':
    S.run_main(main)
