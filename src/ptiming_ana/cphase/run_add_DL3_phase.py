# This script is used to submit jobs for adding phase information to DL3 data files.
# It can either process all runs in a specified directory or a specific run number provided as an argument (-r)

import os
import re
import subprocess
import argparse
from pathlib import Path #(LBZ)
import logging #(LBZ)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') #(LBZ)


def find_run_files(path,file_format=".fits"): #(LBZ)

    """
    Searches for files with a certain format containing the word 'run' in their names within a specified directory and its subdirectories.

    Args:
        path (str): The root directory path to search for files.
        file_format (str, optional): The file extension to search for. Defaults to '.fits'.

    Returns:
        list: A list of paths to files that match the criteria.

    Example:
        >>> root_directory = '/home/zaourat/Documents'
            fits_files = find_run_files(root_directory,'.h5')
            for fits_file in fits_files:
                print(fits_file)
    """


    fits_files = []

    for root, dirs, files in os.walk(path) :
        for file in files :
            if file.endswith(file_format) and 'run' in file.lower():
                fits_files.append(os.path.join(root, file))
    if not fits_files:
        logging.warning(f"No .fits files containing 'run' found in {path}")
    else:
        logging.info(f"Found {len(fits_files)} {file_format} file(s) containing 'run' in {path}")
    return fits_files

# root_directory = '/home/zaourat/Documents'
# fits_files = find_run_files(root_directory)
# for fits_file in fits_files:
#     print(fits_file)

#dir_input_DL3 = "/fefs/aswg/workspace/cyann.buisson/202506_CataCrab/data/dl3/DL3_index/lstchain/0.10.18/std/ghcut0.7_thetacut0.2/"
#dir_input_DL3 = "/fefs/aswg/workspace/cyann.buisson/202506_CataCrab/data/dl3/DL3_index/lstchain/0.10.18/std/gheff0.7_thetacont0.7"
#dir_input_DL3 = "/fefs/aswg/workspace/vincent.poireau/Crab/lappana/dl3/DL3_index/lstchain/0.10.18/std/gheff0.9_thetacont0.9/"
#dir_input_DL3 = "/fefs/aswg/workspace/vincent.poireau/Crab/lappana/dl3/DL3_index/lstchain/0.10.18/std/gheff0.9_alphacont0.9/"
#dir_input_DL3 = "/fefs/aswg/workspace/vincent.poireau/Crab/lappana/dl3/DL3_index/lstchain/0.10.18/std/gheff0.7_alphacont0.7/"
#dir_input_DL3 = "/fefs/aswg/workspace/guillaume.grolleron/project/LSTanalysis/Crab-pulsar/out/dl3/DL3_index/gammaLearn/0.14.3/inference_cleaning_mc_star_lstchain_mask_run_average_maxdist_time_0/0.14.3/"
#dir_input_DL3 = "/fefs/aswg/workspace/guillaume.grolleron/project/LSTanalysis/Crab-gscan-timing-studies/out/dl3/DL3_index/gammaLearn/0.14.3/inference_cleaning_mc_star_lstchain_mask_run_average_maxdist_time_0/0.14.3"
#dir_output_DL3 = "/fefs/aswg/workspace/vincent.poireau/Crab/gheff0.9_thetacont0.9/"
#dir_output_DL3 = "/fefs/aswg/workspace/vincent.poireau/Crab/gheff0.9_alphacont0.9/"
#dir_output_DL3 = "/fefs/aswg/workspace/vincent.poireau/Crab/gheff0.7_alphacont0.7/"

# The folder path that contains the runs in a .fits format (output of Edna's dashboard)
############################################################################################################################
#runs_folder_name='Data_quality_Crab_gheff0.9_thetacont0.7_zmin0.0_zmax75.0_firstdate20200114_lastdate20251129'
runs_folder_name='dl3_lstchain_gheff0.91_alphacont0.91_lappana'
pulsar_name= 'Crab'
gheff_cut= 0.91 # (0.7 or 0.9)
thetacont= 0.91 # (0.7 or 0.9)
############################################################################################################################


# The workspace path 
workspace_crab_path= '/fefs/aswg/workspace/lina.bretonzaourat/crab_analysis/src_dep_analysis'
script_dir = Path(__file__).resolve().parent # LBZ_280826
phase_script = script_dir / "add_DL3_phase.sh" # LBZ_280826

file_ephem = "/fefs/aswg/workspace/lina.bretonzaourat/PulsarTimingAnalysis/src/ptiming_ana/cphase/Crab.gro"

# Extract the pulsar name to create a folder named as the pulsar 
match_name = re.search(r"Data_quality_([a-zA-Z]+)", runs_folder_name)
if match_name:
    pulsar_name = match_name.group(1)
    print('-'*50)
    print(f"Extracted pulsar name: {pulsar_name}")
    # Create a directory for the pulsar results
    output_path_dir= f"{workspace_crab_path}/data/dl3/dl3_lstchain_gheff0.91_alphacont0.91_lappana/phased"
    pulsar_path_dir = os.path.join(output_path_dir, pulsar_name)
    os.makedirs(pulsar_path_dir, exist_ok=True)
#     print(f"Directory '{pulsar_path_dir}' created successfully.")
else:
     print("No pulsar name found in the string.")
     output_path_dir= f"{workspace_crab_path}/data/dl3/dl3_lstchain_gheff0.91_alphacont0.91_lappana/phased"
     os.makedirs(output_path_dir, exist_ok=True)
 
#dir_input_DL3 = f"/{workspace_crab_path}/data/raw/DL3/{pulsar_name}/gheffcut_{gheff_cut}/thetacont_{thetacont}/{runs_folder_name}"
dir_input_DL3 = f"{workspace_crab_path}/data/dl3/dl3_lstchain_gheff0.91_alphacont0.91_lappana"
#gammalearn 
#dir_input_DL3= "/fefs/aswg/workspace/guillaume.grolleron/project/LSTanalysis/Crab-GL-perf/DVR/out/dl3"

dir_output_DL3 =  f"{workspace_crab_path}/data/dl3/dl3_lstchain_gheff0.91_alphacont0.91_lappana/phased/crab/gheffcut_{gheff_cut}/thetacont_{thetacont}/{runs_folder_name}_phased"
#dir_output_DL3 = f"/{workspace_crab_path}/data/processed/DL3/Phased_pulsars/{pulsar_name}/gheffcut_{gheff_cut}/thetacont_{thetacont}/{runs_folder_name}_phased"
#gammalearn
#dir_output_DL3="/fefs/aswg/workspace/lina.bretonzaourat/workspace/Crab_analysis_sourceindep/dl3_gammalearn/dl3_phased"

file_ephem = "/fefs/aswg/workspace/lina.bretonzaourat/PulsarTimingAnalysis/src/ptiming_ana/cphase/Crab.gro"

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_number', dest='run_number', help='Add the phases only for the specified run')
parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode, only valid if the option -r')
args = parser.parse_args()

# If we do not precise the run number, we will run the script for all runs in the input directory
if not args.run_number:

    # Move previous job output files to another directory
    if not os.path.exists("out/out_phase_job/previous_jobs"):
        os.makedirs("out/out_phase_job/previous_jobs")
    cmd = "mv out/*.out out/out_phase_job/previous_jobs/"
    subprocess.run(cmd, shell=True)

    #pattern = re.compile(r"dl3_LST-1\.Run(\d+)\_gh0.7_th0.7_nsb0.38_ring-wobble.fits$")
    pattern= re.compile(r"dl3_LST-1\.Run(\d+)(?:_(.*?))?\.fits$") # permet d'utiliser tous les formats de fichiers de runs avec ou sans les paramètres
    print(f'pattern: {pattern}')
    
    #Search all the files .fits in all the sub folders 
    #fits_files = list(Path(dir_input_DL3).rglob("*.fits")) #(LBZ)
    fits_files = [file for file in Path(dir_input_DL3).rglob("*.fits") if 'run' in file.name.lower()] # (LBZ)
    print("fit_file",fits_files)
    #fits_files = find_run_files(dir_input_DL3,file_format='.fits')
    
    for fits_file in fits_files:
        filename = fits_file.name  
        print('filename:',filename)
        match = pattern.match(filename)
        #print(match)
        if match:
            run_number = match.group(1)
            print(f'Run number: {run_number}')
            parameters= match.group(2) # Stack tout ce qui est après le numéro du run dans le nom du Run (.fits)
            parameters_list= parameters.split('_') if parameters else []
            print(parameters_list)
            print(f'params {parameters}')
            print('-'*50)
            output_file_option = "out/slurm-DL3-%j-" + run_number + ".out"
            # Utiliser le chemin complet du fichier
            cmd = ["sbatch", "-o", output_file_option, "add_DL3_phase.sh", dir_input_DL3, file_ephem, dir_output_DL3, run_number]
            subprocess.run(cmd)

    # Create the output directory if it does not exist
    if not os.path.exists(dir_output_DL3):
        os.makedirs(dir_output_DL3) 

    # # Copy index files to the new output directory
    # cmd = ["cp", dir_input_DL3 + "/obs-index.fits.gz", dir_output_DL3  + "/."]
    # subprocess.run(cmd)
    # #print(cmd)
    # cmd = ["cp", dir_input_DL3 + "/hdu-index.fits.gz", dir_output_DL3  + "/."]
    # subprocess.run(cmd)

    ######################################################################### (LBZ) waiting the creation of the phased files to build the DL3 files ?? 
    # print("Creating phased DL3 index files...")  # Avant
    # cmd = ["lstchain_create_dl3_index_files", "--input-dl3-dir", dir_output_DL3, 
    #    "--file-pattern", "dl3*.fits", "--overwrite"]
    # try:
    #     result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    #     print("DL3 index files created successfully")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error creating DL3 index files:\n{e.stderr}")
    ####################################################################### (LBZ)

# If we precise the run number, we will run the script only for this run
if args.run_number:
    run_number = args.run_number
    if len(run_number) == 4:
        run_number = "0" + run_number
    print(f'Run number: {run_number}')
    print('-'*50)
    if args.interactive:
        cmd = [str(phase_script), dir_input_DL3, file_ephem, dir_output_DL3, run_number] # LBZ_280826
        print("interactive_cmd_done")
    else:
        output_file_option = "slurm-DL3-%j-" + run_number + ".out"
        cmd = ["sbatch", "-o", output_file_option, str(phase_script), dir_input_DL3, file_ephem, dir_output_DL3, run_number] # LBZ_280826
    subprocess.run(cmd, check=True) # LBZ_280826

