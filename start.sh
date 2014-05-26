#!/bin/bash

#Star the simulation

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
py=$(which python)
prog=$DIR"/mininet/builder.py"

#elevate to root
[ "$UID" -eq 0 ] || exec sudo "$0" "$@"

ulimit -n 100000
"$py" "$prog" "$@"

#sudo mn --clean

#sudo pkill -9 -f "$HOME"
