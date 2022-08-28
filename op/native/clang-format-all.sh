#!/bin/sh
find -iname "*.h" -o -iname "*.cpp" -o -iname "*.cuh" -o -iname "*.cu" | xargs clang-format -i
