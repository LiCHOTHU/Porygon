#!/bin/bash

# Directory paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${BASE_DIR}/data/real/raw"
OUTPUT_DIR="${BASE_DIR}/data/real"
SCRIPT_PATH="${BASE_DIR}/scripts/process_real.py"

mkdir -p ${OUTPUT_DIR}

file_names=(
    "apple_in_red_bowl.hdf5"
    "avocado_in_blue_bowl.hdf5"
    "bell_pepper_on_plate.hdf5"
    "carrot_on_plate.hdf5"
    "grapes_in_bowl.hdf5"
    "lemon_in_bowl.hdf5"
)

# Process each HDF5 file
for file_name in ${file_names[@]}; do
    uv run ${SCRIPT_PATH} ${DATA_DIR}/${file_name} ${OUTPUT_DIR}/${file_name} --allow-overwrite
done



#  python scripts/process_mimicgen.py \
#     --hdf5_path /home/awilcox31/imitation/data/mimicgen/core/square_d1.hdf5 \
#     --output_dir /home/awilcox31/imitation/data/mimicgen/test/square_d1.hdf5 \
#     --depth --n 3 --allow_overwrite

