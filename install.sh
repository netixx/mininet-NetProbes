#!/bin/bash

if [ -z "$1" ]
then
    CPU="x86_64"
else
    CPU="$1"
fi

ECHO_PREFIX="**install.sh**"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIBS_DIR_NAME="libs"
BIN_DIR_NAME="bin"
LIBS_DIR="$DIR/$LIBS_DIR_NAME"
BIN_DIR="$DIR/$BIN_DIR_NAME"

mkdir "$BIN_DIR"

YAZ_DIR="yaz-master"
IGI_DIR="igi-ptr-2.1"
ASSOLO_DIR="assolo"
ABING_DIR="abing_2.2.0"
SPRUCE_DIR="spruce-0.3"
TRACEROUTE_DIR="traceroute-2.0.18"

LN_SPRUCE_BIN_DIR="spruce"
LN_IGI_BIN_DIR="igi"


BIN_DIRS_MK=("$LN_SPRUCE_BIN_DIR" "$LN_IGI_BIN_DIR")
EXECS_FILES=("$IGI_DIR/ptr-client" "$IGI_DIR/ptr-server" "$SPRUCE_DIR/spruce_rcv" "$SPRUCE_DIR/spruce_snd")
LNS_FILES=("$LN_IGI_BIN_DIR" "$LN_IGI_BIN_DIR" "$LN_SPRUCE_BIN_DIR" "$LN_SPRUCE_BIN_DIR")

EXECS_DIR=("$YAZ_DIR/yaz" "$ASSOLO_DIR/Bin/$CPU" "$ABING_DIR/Bin/$CPU" "$TRACEROUTE_DIR/traceroute/traceroute")
LNS_DIR=("yaz" "assolo" "abing" "traceroute")

LIBS_SRC=("$YAZ_DIR" "$IGI_DIR" "$ASSOLO_DIR" "$ABING_DIR" "$SPRUCE_DIR" "$TRACEROUTE_DIR")

echo "$ECHO_PREFIX Making all libraries in $LIBS_DIR"
for lib in "${LIBS_SRC[@]}"
do
    cd "$LIBS_DIR/$lib"
    echo "$ECHO_PREFIX Making in $LIBS_DIR_NAME/$lib"
    make clean
    ./configure
    make
done

cd "$BIN_DIR"

echo "$ECHO_PREFIX Linking libraries to $BIN_DIR"
for (( i=0; i<${#EXECS_DIR[@]}; i++ ))
do
    echo "$ECHO_PREFIX Linking $LIBS_DIR_NAME/${EXECS_DIR[$i]} to $BIN_DIR_NAME/${LNS_DIR[$i]}"
    ln -sfn "../$LIBS_DIR_NAME/${EXECS_DIR[$i]}" "${LNS_DIR[$i]}"
done
for dir in "${BIN_DIRS_MK[@]}"
do
    mkdir $dir
done

for (( i=0; i<${#LNS_FILES[@]}; i++ ))
do
    cd "${LNS_FILES[$i]}"
    echo "$ECHO_PREFIX Linking $LIBS_DIR_NAME/${EXECS_FILES[$i]} to $BIN_DIR_NAME/${LNS_FILES[$i]}/"
    ln -sfn "../../$LIBS_DIR_NAME/${EXECS_FILES[$i]}" .

    cd "$BIN_DIR"
done

cd "$DIR"

mkdir "checks"
