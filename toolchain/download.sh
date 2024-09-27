#!/bin/bash

# Download the toolchain
if [ ! -f nds64le-linux-glibc-v5d.txz ]; then
    wget https://github.com/andestech/Andes-Development-Kit/releases/download/ast-v5_3_0-release-linux/nds64le-linux-glibc-v5d.txz
fi

# Extract the toolchain
if [ ! -d nds64le-linux-glibc-v5d ]; then
    tar -xf nds64le-linux-glibc-v5d.txz
fi