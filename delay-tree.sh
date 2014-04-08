#!/bin/bash

#Start the simulation

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog=$DIR"/start.sh"

sudo "$prog" --vars dtime=10000ms delay=500ms --start 10 --no-netprobes --topo delay-tree
