#!/bin/bash

#SBATCH --time=12:00:00           # walltime
#SBATCH --ntasks=32               # number of processor cores (i.e. tasks)
#SBATCH --nodes=1                 # number of nodes
#SBATCH --mem=2048G               # memory per compute node
#SBATCH -J "HHBlits"              # job name
#SBATCH --output=./slurm_out/HHblits_%J.out 

module load parallel

TMPDIR=/dev/shm/$SLURM_JOB_ID

# Called when job is cancelled
cleanup_scratch() {
    rm -rfdv $TMPDIR
    exit 1
}
trap 'cleanup_scratch' TERM

mkdir $TMPDIR

export BASE="/fslhome/bryceeh/compute/thesis"

# Copy database to TMPDIR
cp -v ${BASE}/db/bfd_metaclust/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt* ${TMPDIR}
export D="${TMPDIR}/bfd_metaclust"

export HHBLITS="${BASE}/hh-suite/build/bin/hhblits"
export OPTIONS="-cpu 1 -n 1 -aliw 32000 -B 10000"

function align {
    OA3M="${BASE}/data/bfd/a3m/$1"
    [ -f ${OA3M} ] && return
    [ $(less ${OA3M} | wc -l) -gt 2000 ] && return

    I="${BASE}/data/uniref/a3m/$1"
    O="${BASE}/data/bfd/hhr/$1"
    ${HHBLITS} -i $I -d $D -o $O -oa3m $OA3M ${OPTIONS} 
}

export -f align
PWD=$(pwd)
cd ${BASE}/data/seq
list=$(find . -type f | shuf)
cd $PWD
parallel -j $SLURM_NTASKS align ::: $list

rm -rfdv $TMPDIR
