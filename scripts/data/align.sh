#!/bin/bash

BASE="/fslhome/bryceeh/compute/thesis"

HHBLITS="${BASE}/hh-suite/build/bin/hhblits"

I="${BASE}/data/seq/$1"
D="${BASE}/db/UniRef30_2020_03/UniRef30_2020_03"
O="${BASE}/data/hhr/$1"
OA3M="${BASE}/data/a3m/$1"


OPTIONS="-cpu 8 -n 3 -e 1 -mact 0.0 -aliw 32000 -b 1 -B 10000 -z 1 -Z 10000"

${HHBLITS} -i $I -d $D -o $O -oa3m $OA3M ${OPTIONS}
