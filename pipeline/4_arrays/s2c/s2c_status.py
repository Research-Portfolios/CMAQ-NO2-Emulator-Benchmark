"""Prints the clock, the free disk and the available physical memory in decimal GB (read only)."""
import ctypes
import datetime
import shutil

class MEMSTAT(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

def main():
    now = datetime.datetime.now().astimezone()
    print('clock %s' % now.strftime('%Y-%m-%d %H:%M:%S %z'))
    du = shutil.disk_usage('.')
    print('work disk free %.2f GB of %.2f GB (bytes %d)' % (du.free / 1000000000.0, du.total / 1000000000.0, du.free))
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    print('memory: %.2f GB available of %.2f GB' % (m.ullAvailPhys / 1000000000.0, m.ullTotalPhys / 1000000000.0))
if __name__ == '__main__':
    main()
