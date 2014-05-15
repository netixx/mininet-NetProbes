#!/bin/bash

#Start the tree simulation @ data/delay-tree.json

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog=$DIR"/start.sh"

#"$prog" --no-command --vars dtime=100000ms delay=300ms --start 30 --topo delay-tree --monitor usages/monitor.txt
"$prog" --command "$DIR/start-probe.sh {commandOpts}" --vars dtime=1000000ms delay=20ms --auto-start-events 20 --topo delay-tree
