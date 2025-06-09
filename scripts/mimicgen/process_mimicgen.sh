#!/bin/bash

# Directory paths
BASE_DIR="/storage/home/hcoda1/1/awilcox31/vast/imitation"
DATA_DIR="${BASE_DIR}/data/mimicgen/core"
OUTPUT_DIR="${BASE_DIR}/data/mimicgen/core_depth_abs"
SCRIPT_PATH="${BASE_DIR}/scripts/process_mimicgen.py"

mkdir -p ${OUTPUT_DIR}

file_names=(
    "coffee_d1.hdf5" 
    "square_d1.hdf5" 
    "threading_d1.hdf5"
    "stack_d1.hdf5"
    "stack_three_d0.hdf5"
    "stack_three_d1.hdf5"
    "three_piece_assembly_d0.hdf5"
    "three_piece_assembly_d1.hdf5"
)

# Process each HDF5 file
for file_name in ${file_names[@]}; do
    echo "Submitting job for $file_name..."
    sbatch slurm/run_rtx6000.sbatch uv run ${SCRIPT_PATH} --hdf5_path ${DATA_DIR}/${file_name} --output_dir ${OUTPUT_DIR}/${file_name} --depth
done

echo "All jobs submitted!" 


#  python scripts/process_mimicgen.py \
#     --hdf5_path /home/awilcox31/imitation/data/mimicgen/core/square_d1.hdf5 \
#     --output_dir /home/awilcox31/imitation/data/mimicgen/test/square_d1.hdf5 \
#     --depth --n 3 --allow_overwrite

