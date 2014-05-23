#!/bin/bash
#usage flood-flat output_file nhosts subnetlength

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ "$3" -gt 8 ]]
then
echo "Subnet length must be <= 8"
exit
fi

OUTPUT="$DIR/../mininet/data/$1.json"
NHOSTS="$2"
PREFIX_LENGTH="$3"
NHOSTPERPREFIXES="$(expr $(( 1 << $3 )) - 2)"

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
vprefix="0"
prefixprefix="10.0"
#ip="$(getprefix "$prefixprefix" "$vprefix" "$(expr 1 % "$NHOSTPERPREFIXES")")"
hosts=""
#$(gethost "${hostname}1" "$ip")"
links=""
#$(getlink "$switchname" "${hostname}1")"
sep=""
for i in $(seq 0 $(expr $NHOSTS - 1))
do
    hnum="$(expr $i + 1)"
    lip=$(expr "$i" % "$NHOSTPERPREFIXES")
    if  [ "$lip" -eq 0 ]
    then
        vprefix=$(expr "$vprefix" + 1)
    fi
    ip="$(getprefix "$prefixprefix" "$vprefix" "$(expr "$lip" + 1)")"
    hosts="$hosts$sep$(gethost "$hostname$hnum" "$ip")"
    links="$links$sep$(getlink "$switchname" "$hostname$hnum")"
    sep=","
done

out=$(getjson "$(gethosts "$hosts")" "$(getswitch "$switchname")" "$(getlinks "$links")")

echo -e "$out" > $OUTPUT
