#!/bin/bash

# Spécifie le compte (ou allocation) à utiliser pour ce job
#SBATCH -A aswg

# Duration of the job (short,long)
#SBATCH -p short

# Name of the job 
#SBATCH -J sed_ana

# Memory allocation
#SBATCH --mem=20g

# Remove locked memory limit
ulimit -l unlimited

# Remove stack size limit
ulimit -s unlimited

# Display all the limits 
ulimit -a

CONFIG_PATH="$1"
PEAK="$2"
OUTPUT_DIR="$3"

echo "=========================================="
echo "Spectral Analysis (SED) Job"
echo "Config: $CONFIG_PATH"
echo "Peak: ${PEAK:-from config}"
echo "Output: ${OUTPUT_DIR:-from config}"
echo "=========================================="

# Activate virtual environment if needed
# source activate pulsar-lst1

# Run the spectral analysis
if [ -z "$PEAK" ] && [ -z "$OUTPUT_DIR" ]; then
    python3 run_sed.py --config_path="$CONFIG_PATH"
elif [ -z "$OUTPUT_DIR" ]; then
    python3 run_sed.py --config_path="$CONFIG_PATH" --peak="$PEAK"
else
    python3 run_sed.py --config_path="$CONFIG_PATH" --peak="$PEAK" --output_dir="$OUTPUT_DIR"
fi
