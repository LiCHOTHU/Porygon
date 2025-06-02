#!/bin/bash

# Directory paths
BASE_DIR="/storage/home/hcoda1/1/awilcox31/vast/quest_mimicgen"
DATA_DIR="${BASE_DIR}/data/mimicgen/core"
OUTPUT_DIR="${BASE_DIR}/data/mimicgen/core_depth_abs_3"
SCRIPT_PATH="${BASE_DIR}/scripts/process_mimicgen.py"

mkdir -p ${OUTPUT_DIR}

file_names=("coffee_d1.hdf5" "square_d1.hdf5" "threading_d1.hdf5")

# Process each HDF5 file
for file_name in ${file_names[@]}; do
    echo "Submitting job for $file_name..."
    sbatch slurm/run_rtx6000_premium.sbatch python ${SCRIPT_PATH} --hdf5_path ${DATA_DIR}/${file_name} --output_dir ${OUTPUT_DIR}/${file_name} --depth
done

echo "All jobs submitted!" 


#  python scripts/process_mimicgen.py --hdf5_path /storage/home/hcoda1/7/mghanem8/p-agarg35-0/codes/quest_v0/data/mimicgen/core_depth/coffee_d1.hdf5 --output_dir /storage/home/hcoda1/7/mghanem8/p-agarg35-0/codes/quest_v0/data/mimicgen/core_depth_abs/coffee_d1.hdf5 --depth

