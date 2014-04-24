#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

OUTPUT="$DIR/../mininet/data/$1.json"
NHOSTS="$2"
NHOSTPERPREFIXES="$3"

hostname="h"
linkname="l"
switchname="s1"
linkname="l"

getjson() {
    echo "{
$1,
$2,
$3
}
"

}

getswitch() {
    echo "    \"switches\" : [
        {
            \"name\" : \"$1\"
        }
    ]
"

}
getlinks() {
    echo "    \"links\" : [
$1
    ]
"

}

getlink() {
    echo "
        {
			\"hosts\" : [\"$1\", \"$2\"]
		}
"
}

gethosts() {
    echo "    \"hosts\" : [$1
    ]
"
}


gethost() {
    echo "
        {
            \"name\" : \"$1\",
            \"options\" : {
                \"ip\" : \"$2\"
            }
        }
"

}

getprefix() {
    echo "$1.$2.$3"

}
vprefix="1"
prefixprefix="10.0"
ip="$(getprefix "$prefixprefix" "$vprefix" "$(expr 1 % "$NHOSTPERPREFIXES")")"
hosts="$(gethost "${hostname}1" "$ip")"
links="$(getlink "$switchname" "${hostname}1")"

for i in $(seq 2 $NHOSTS)
do
    lip=$(expr "$i" % "$NHOSTPERPREFIXES")
    if  [ "$lip" -eq 0 ]
    then
        vprefix=$(expr "$vprefix" + 1)
    fi
    ip="$(getprefix "$prefixprefix" "$vprefix" "$(expr "$lip" + 1)")"
    hosts=$hosts",$(gethost "$hostname$i" "$ip")"
    links=$links",$(getlink "$switchname" "$hostname$i")"
done

out=$(getjson "$(gethosts "$hosts")" "$(getswitch "$switchname")" "$(getlinks "$links")")

echo -e "$out" > $OUTPUT
