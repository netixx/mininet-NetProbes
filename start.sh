#!/bin/bash

#Star the simulation

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
py=$(which python)
prog=$DIR"/mininet/builder.py"

sudo "$py" "$prog" "$@"

sudo mn --clean
