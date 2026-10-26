# FILE MAP -- shell.py
# WHAT: the file Python runs for `python -m minisgl.shell`. Same as __main__.py but starts an
#       interactive terminal chat (run_shell=True) instead of an HTTP server.
# ELI5: instead of a front desk with a door to the street, the manager sits at a table and
#       chats with you directly. The same kitchen is behind it.
# HOW IT DIFFERS: server/args.py shrinks the setup in shell mode (one request at a time, quiet
#       logs), and server/api_server.py's `shell()` function does the typing loop.
# Tag legend: see zero-to-sglang/eli5/README.md
from .server import launch_server

if __name__ == "__main__":
    launch_server(run_shell=True)
