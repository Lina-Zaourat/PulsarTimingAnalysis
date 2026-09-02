#!/bin/bash
#SBATCH -A aswg
#SBATCH -p long
#SBATCH -J add_DL2_phase_$i
#SBATCH --mem=64g

ulimit -l unlimited
ulimit -s unlimited
ulimit -a


echo
echo
echo ------------------------------------------
echo Running add_DL2_phase_table for:
echo Input DL2: $1
echo Ephemeride file: $2
echo ------------------------------------------
echo
echo


add_DL2_phase_table --in-file=$1 --ephem=$2


