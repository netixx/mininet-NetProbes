#!/bin/bash

#Start the simulation @data/delay-flat.json

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog=$DIR"/start.sh"

"$prog" --command "$DIR/start-probe.sh {commandOpts}" --force-x --vars dtime=100000ms delay=500ms --start 20 --topo delay-flat