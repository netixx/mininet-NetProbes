#!/bin/bash

#Start the simulation
set -m
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog=$DIR"/start.sh"

"$prog" -x --vars dtime=100000ms delay=500ms --start 20 --topo delay-flat &

sleep 5

xterm -e bash -c "$DIR/root-probe.sh --add-prefix 10.0.0.1/30" &

fg %1 > /dev/null
