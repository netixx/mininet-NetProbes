#!/bin/bash

#Start the tree simulation @ data/delay-tree.json

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog=$DIR"/start.sh"

"$prog" --no-command --vars dtime=100000ms delay=300ms --start 30 --topo delay-tree --monitor usages/monitor.txt
#"$prog" --command "$DIR/start-probe.sh {commandOpts}" --no-command --vars dtime=100000ms delay=500ms --start 30 --topo delay-tree
