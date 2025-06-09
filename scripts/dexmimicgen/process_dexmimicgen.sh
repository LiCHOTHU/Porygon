#!/bin/bash

# Directory paths
BASE_DIR="/storage/home/hcoda1/1/awilcox31/vast/imitation"
BASE_DIR="/storage/home/hcoda1/1/awilcox31/vast/imitation/"
# DATA_DIR="${BASE_DIR}/data/dexmimicgen/generated"
# OUTPUT_DIR="${BASE_DIR}/data/dexmimicgen/processed"
# SCRIPT_PATH="${BASE_DIR}/scripts/process_dexmimicgen.py"

# mkdir -p ${OUTPUT_DIR}

# file_names=(
#     "two_arm_coffee.hdf5"
#     "two_arm_pouring.hdf5"
# )

# # Process each HDF5 file
# for file_name in ${file_names[@]}; do
#     echo "Submitting job for $file_name..."
#     sbatch slurm/run_rtx6000.sbatch python ${SCRIPT_PATH} --hdf5_path ${DATA_DIR}/${file_name} --output_dir ${OUTPUT_DIR}/${file_name} --depth
# done

# echo "All jobs submitted!" 


python scripts/process_dexmimicgen.py \
    --hdf5_path ${BASE_DIR}/data/dexmimicgen/generated/two_arm_coffee.hdf5 \
    --output_dir ${BASE_DIR}/data/dexmimicgen/test/two_arm_coffee.hdf5 \
    --depth --n 3 --allow_overwrite

