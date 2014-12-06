#!/bin/bash

#Start the tree simulation with bandwidth detection

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

"$DIR/../run-bandwidth.sh" "$DIR/vars"

