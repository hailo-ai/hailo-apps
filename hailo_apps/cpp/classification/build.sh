#!/bin/bash

declare -A COMPILER=( [x86_64]=/usr/bin/gcc 
                      [aarch64]=/usr/bin/aarch64-linux-gnu-gcc 
                      [armv7l]=/usr/bin/arm-linux-gnueabi-gcc )

for ARCH in x86_64
do
    echo "-I- Building ${ARCH}"
    mkdir -p build/${ARCH}
    CXX=g++-9 cmake -H. -Bbuild/${ARCH} -DCMAKE_C_COMPILER=${COMPILER[${ARCH}]}
    cmake --build build/${ARCH}
done
if [[ -f "hailort.log" ]]; then
    rm hailort.log
fi
