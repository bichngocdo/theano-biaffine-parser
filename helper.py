import importlib
import sys

if __name__ == '__main__':
    package = 'dense.models.' + sys.argv[1]
    command = '.' + sys.argv[2]
    module = importlib.import_module(command, package)
    print module.main(sys.argv[3:])
