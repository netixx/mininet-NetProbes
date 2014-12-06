#!/bin/bash

#Start the tree simulation @ data/delay-tree.json

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WA_DIR="$DIR/watchers"

#PARAMS="exp-dep-noop.sh exp-dep-op.sh"
PARAMS="exp-dep-noop-9x2.sh exp-dep-op-9x2.sh"

for param in ${PARAMS}
do
echo ">>>> Start set $param at $(date '+%m-%d_%T')"

mkdir "/tmp/matplotlib-$USER"
sudo mn -c

"$DIR/explore-depth.sh" "$param"

now="$(date '+%m-%d_%T')"

echo ">>>> Run done at $now, saving output and logs"
logs="$WA_DIR/logs/watchers_$now"
outputs="$WA_DIR/output/watchers_$now"
mkdir "$logs"
mkdir "$outputs"
out="$WA_DIR/watchers_$now.json"
outgr="$WA_DIR/watchers_$now.pdf"

#sudo mv "$WA_DIR/watchers-link.pdf" "$WA_DIR/$now.pdf"
sudo mv "$WA_DIR/logs/"*.log "$logs/"
sudo mv "$WA_DIR/output/"*.json "$outputs/"
sudo mv "$WA_DIR/watchers.json" "$out"


if [[ -f "$out" ]]
then
echo ">>>> Making graphics from $out to $outgr"
#make graph
python "mininet/watcher_delay.py" --output "$out" --graphs "$outgr"
fi

echo ">>>> Set done"
done
