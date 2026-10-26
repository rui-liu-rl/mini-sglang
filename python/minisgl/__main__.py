# FILE MAP -- __main__.py
# WHAT: the file Python runs for `python -m minisgl`. It does exactly one thing: open the
#       restaurant (launch_server). All flags are read from the command line by server/args.py.
# ELI5: this is the OPEN sign on the door.
# Tag legend: see zero-to-sglang/eli5/README.md
from .server import launch_server

# GOTCHA: this assert makes it an error to import this module; it is meant to be run, not imported.
assert __name__ == "__main__"

launch_server()
