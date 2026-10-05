"""python -m ego <command> [options]"""

import importlib
import sys

from ego.commands import COMMANDS


def usage():
    print("usage: python -m ego <command> [options]\n\ncommands:")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:<15} {desc}")
    print("\nRun `python -m ego <command> --help` for a command's options.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        usage()
        return 0
    name = sys.argv[1]
    if name not in COMMANDS:
        print(f"unknown command {name!r}\n")
        usage()
        return 2
    module = importlib.import_module(f"ego.commands.{COMMANDS[name][0]}")
    sys.argv = [f"python -m ego {name}"] + sys.argv[2:]
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
