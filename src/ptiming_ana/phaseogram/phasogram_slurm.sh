#!/bin/bash

# Spécifie le compte (ou allocation) à utiliser pour ce job. Ici, le compte est aswg
#SBATCH -A aswg

# Duration of the job (short,long)
#SBATCH -p long

# Name of the job 
#SBATCH -J phasogram_run_$2

# Memor allocation
#SBATCH --mem=20g

# Remove locked memory limit
ulimit -l unlimited

# Remove stack size limit
ulimit -s unlimited

# Display all the limits 
ulimit -a

CONFIG_PATH="$1"
RUN_NUMBER="$2"
OUTPUT_DIR="$3"

echo "=========================================="
echo "Phasogram Analysis for Run: $RUN_NUMBER"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python3 phasogram.py \
    --config_path="$CONFIG_PATH" \
    --run_number="$RUN_NUMBER" \
    --output_dir="$OUTPUT_DIR"
