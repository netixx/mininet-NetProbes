#!/bin/bash

#Start the simulation

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog="$HOME/netprobes/start.sh"

"$prog" --commander -id root --watcher delay "$@"
