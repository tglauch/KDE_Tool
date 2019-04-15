#!/bin/bash


source '/home/software/root/install/6.14-04/bin/thisroot.sh'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
python "$SDIR/ps_kde/bw_cv.py" $@
