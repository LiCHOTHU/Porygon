import os

import numpy as np
np.set_printoptions(suppress=True)

from torch.utils.data import Dataset, ConcatDataset

import imitation.utils.file_utils as FileUtils
import imitation.utils.obs_utils as ObsUtils
from imitation.utils.dataset import SequenceDataset

# task_name_to_task_files = {
#     "coffee": ["coffee_d1.hdf5"],
#     "coffee_preparation": ["coffee_preparation_d1.hdf5"],
#     "hammer_cleanup": ["hammer_cleanup_d1.hdf5"],
#     "kitchen": ["kitchen_d1.hdf5"],
#     "mug_cleanup": ["mug_cleanup_d1.hdf5"],
#     "square": ["square_d1.hdf5"],
#     "stack": ["stack_d1.hdf5"],
#     "stack_three": ["stack_three_d1.hdf5"],
#     "threading": ["threading_d1.hdf5"],
#     "three_piece_assembly": ["three_piece_assembly_d1.hdf5"]
# }

def task_name_to_task_files(task_name):
    return [f"{task_name}.hdf5"]

def build_single_task_dataset(data_prefix,
                              task_name, 
                              seq_len, 
                              frame_stack,
                              shape_meta,
                              obs_seq_len=1, 
                              extra_obs_modality=None,
                              load_obs=True,
                              n_demos=None,
                              dataset_keys=('actions',)
                              ):
    dataset_path = os.path.join(data_prefix, 'mimicgen', 'core_depth_abs')

    obs_modality = {
        'rgb': list(shape_meta['observation']['rgb'].keys()),
        'depth': list(shape_meta['observation']['depth'].keys()),
        'low_dim': list(shape_meta['observation']['lowdim'].keys())
    }

    if extra_obs_modality is not None:
        for key in extra_obs_modality:
            obs_modality[key] = obs_modality[key] + extra_obs_modality[key]

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    task_files = [os.path.join(dataset_path, file) for file in task_name_to_task_files(task_name)]
    
    # Process each file and create a dataset
    task_datasets = []
    total_demos = 0
    total_sequences = 0
    
    for file_path in task_files:
        task_dataset = get_task_dataset(
            dataset_path=file_path,
            obs_modality=obs_modality,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            load_obs=load_obs,
            frame_stack=frame_stack,
            n_demos=n_demos,
            dataset_keys=dataset_keys,
        )
        dataset = SequenceVLDataset(task_dataset, 0)  # TODO: task_id is always set to 0
        task_datasets.append(dataset)
        total_demos += dataset.n_demos
        total_sequences += dataset.total_num_sequences
    
    # Combine all datasets
    combined_dataset = ConcatDataset(task_datasets)
    
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: MimicGen")
    print(f" # Task: {task_name}")
    print(f" # Files: {len(task_files)}")
    print(f" # demonstrations: {total_demos}")
    print(f" # sequences: {total_sequences}")
    print("=======================================================================\n")

    return combined_dataset

def get_task_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    few_demos=None,
    load_obs=True,
    n_demos=None,
    dataset_keys=None,
):
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    seq_len = seq_len
    filter_key = filter_key
    if load_obs:
        obs_keys = shape_meta["all_obs_keys"]
    else:
        obs_keys = []
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=dataset_keys,
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        obs_seq_length=obs_seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
        few_demos=few_demos,
        n_demos=n_demos,
    )
    return dataset

class SequenceVLDataset(Dataset):
    # Note: task_id should be a string
    def __init__(self, sequence_dataset, task_id):
        self.sequence_dataset = sequence_dataset
        self.task_id = task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_id"] = self.task_id
        return return_dict