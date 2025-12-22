#!/bin/bash

cmake -H. -Bbuild -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$PWD/build/x86_64
cmake --build build

if [[ -f "hailort.log" ]]; then
    rm hailort.log
fi
