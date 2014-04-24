#!/bin/bash
#Start a simulation based of the flood-flat-n json data
#flood-flat-n files are simulation with important number of host in a flat topology
#They can be generated with the generators/flood-flat.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog=$DIR"/start.sh"

"$prog" --command "$DIR/start-probe.sh {commandOpts}" --topo "flood-flat-$1" --monitor "usages/flat-$1.txt"
