#!/bin/bash

# Directory paths
BASE_DIR="/storage/home/hcoda1/7/mghanem8/p-agarg35-0/codes/quest_v0"
DATA_DIR="${BASE_DIR}/data/mimicgen/core"
OUTPUT_DIR="${BASE_DIR}/data/mimicgen/core_depth"
SCRIPT_PATH="${BASE_DIR}/scripts/process_mimicgen.py"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Process each HDF5 file
for hdf5_file in ${DATA_DIR}/*.hdf5; do
    # Extract basename of file
    filename=$(basename "$hdf5_file")
    
    # Submit the job directly
    echo "Submitting job for $filename..."
    sbatch slurm/run_rtx6000.sbatch python ${SCRIPT_PATH} --hdf5_path ${DATA_DIR}/${filename} --output_dir ${OUTPUT_DIR}/${filename} --depth
done

echo "All jobs submitted!" 