#!/bin/bash

# Spécifie le compte (ou allocation) à utiliser pour ce job. Ici, le compte est aswg
#SBATCH -A aswg

# Ordre de priorité des jobs (ici short)
#SBATCH -p short

# Donne un nom au job
#SBATCH -J add_DL3_phase_$i

# Alloue 32 Go de mémoire pour ce job.
#SBATCH --mem=100g


# (IA) Supprime la limite sur la taille de la mémoire verrouillée
ulimit -l unlimited

# (IA) Supprime la limite sur la taille de la pile (stack size), ce qui peut éviter des erreurs de dépassement de pile.
ulimit -s unlimited

# (IA) Affiche toutes les limites actuelles pour vérification.
ulimit -a


echo
echo
echo ------------------------------------------
echo Running add_DL3_phase_table for:
echo Input DL3: $1
echo Ephemeride file: $2
echo Output DL3: $3
echo Run number: $4
echo ------------------------------------------
echo
echo


add_DL3_phase_table --dir=$1/ --ephem=$2 --output=$3/ --run-number=$4

#mv $3/dl3_LST-1.Run$4_pulsar.fits $3/dl3_LST-1.Run$4.fits
