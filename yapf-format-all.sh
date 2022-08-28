#!/bin/sh
find -iname "*.py" | xargs yapf -i
