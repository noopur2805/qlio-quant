#!/usr/bin/env bash
# Build the qlio_ekf pybind11 extension into cpp/.
set -euo pipefail
cd "$(dirname "$0")"
c++ -O3 -march=native -Wall -shared -std=c++17 -fPIC \
    -I/usr/include/eigen3 \
    $(python3 -m pybind11 --includes) \
    qlio_ekf.cpp \
    -o qlio_ekf$(python3-config --extension-suffix)
echo "built: qlio_ekf$(python3-config --extension-suffix)"
