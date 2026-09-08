# This script is used to submit jobs for adding phase information to DL2 data files.
# It can either process all runs in a specified directory or a specific run number provided as an argument (-r)

import os
import re
import subprocess
import argparse
from pathlib import Path

dir_input_DL2 = "/fefs/aswg/workspace/vincent.poireau/Crab/dl2_with_phases/"
file_ephem = "Crab.gro"


# Move previous job output files to another directory
if not os.path.exists("out/previous_jobs"):
    os.makedirs("out/previous_jobs")
cmd = "mv out/*.out out/previous_jobs/"
subprocess.run(cmd, shell=True)

pattern = re.compile(r"^dl2_LST-1.Run\d+\.h5$")
dataset=[]
dataset.append([
    _dataset for _dataset in Path(dir_input_DL2).rglob(f"./*/dl2_LST-1.Run*.h5")
    if _dataset.is_file() and pattern.match(_dataset.name)
    ])
dataset[-1].sort()
print(f"Found {len(dataset[-1])} DL2 data")
pattern = re.compile(r"dl2_LST-1\.Run(\d+)\.h5$")
for i, _dataset in enumerate(dataset[-1]):
    match = pattern.match(_dataset.name)
    run_number = match.group(1)
    print(f"Launching on slurm add_DL2_phase.sh {_dataset} {file_ephem}")
    output_file_option = "out/slurm-DL2-%j-" + run_number + ".out"
    cmd = ["sbatch", "-o", output_file_option, "./add_DL2_phase.sh", str(_dataset), file_ephem]
    subprocess.run(cmd)

