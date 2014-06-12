#!/bin/bash

#Star the simulation

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
py=$(which python)
prog=$DIR"/mininet/builder.py"

#elevate to root
[ "$UID" -eq 0 ] || exec sudo "$0" "$@"

ulimit -n 100000
ulimit -s unlimited
ulimit -c unlimited
ulimit -e unlimited
ulimit -i unlimited
ulimit -l unlimited
#ulimit -p 10
ulimit -q unlimited
ulimit -r unlimited
ulimit -s unlimited
ulimit -u unlimited

#ulimit -a

"$py" "$prog" "$@"

#sudo mn --clean

#sudo pkill -9 -f "$HOME"
