"""Prints the clock, the free disk and the available memory in decimal GB (read only)."""
import datetime
import shutil
import psutil

def main():
    now = datetime.datetime.now().astimezone()
    du = shutil.disk_usage('.')
    vm = psutil.virtual_memory()
    print('clock %s' % now.strftime('%Y-%m-%d %H:%M:%S %z'))
    print('work disk free %.2f GB of %.2f GB (bytes %d)' % (du.free / 1000000000.0, du.total / 1000000000.0, du.free))
    print('memory (psutil available): %.2f GB of %.2f GB' % (vm.available / 1000000000.0, vm.total / 1000000000.0))
if __name__ == '__main__':
    main()
