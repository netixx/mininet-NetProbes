#!/bin/bash

#Start the tree simulation @ data/tree.json

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog=$DIR"/start.sh"

"$prog" --no-command -cc --topo tree
#"$prog" --command "$DIR/start-probe.sh {commandOpts}" --no-command --vars dtime=100000ms delay=500ms --start 30 --topo delay-tree
