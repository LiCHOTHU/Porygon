import os
from hydra.utils import to_absolute_path
from tqdm import trange
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
np.set_printoptions(suppress=True)

from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from transformers import AutoModel, AutoTokenizer, logging


import imitation.utils.file_utils as FileUtils
import imitation.utils.obs_utils as ObsUtils
from imitation.dataset.sequence_dataset import SequenceDataset
from imitation.dataset.sequence_vl_dataset import SequenceVLDataset


def task_name_to_path(task_name):
    return f'{task_name}.hdf5'

benchmarks = {
    "debug_1": [
        "apple_in_red_bowl",
        "avocado_in_blue_bowl",
        "bell_pepper_on_plate",
        "carrot_on_plate",
        "grapes_in_bowl",
        "lemon_in_bowl",
    ],
}

instructions = {
    "apple_in_red_bowl": "pick up the apple and place it in the red bowl",
    "avocado_in_blue_bowl": "pick up the avocado and place it in the blue bowl",
    "bell_pepper_on_plate": "pick up the bell pepper and place it on the plate",
    "carrot_on_plate": "pick up the carrot and place it on the plate",
    "grapes_in_bowl": "pick up the grapes and place them in the bowl",
    "lemon_in_bowl": "pick up the lemon and place it in the bowl",
}
# instructions = {
#     "apple_in_red_bowl": "a photo of a red apple",
#     "avocado_in_blue_bowl": "a photo of an avocado",
#     "bell_pepper_on_plate": "a photo of a red bell pepper",
#     "carrot_on_plate": "a photo of an orange carrot",
#     "grapes_in_bowl": "a photo of green grapes",
#     "lemon_in_bowl": "a photo of a yellow lemon",
# }
# instructions = {
#     "apple_in_red_bowl": "bowl",
#     "avocado_in_blue_bowl": "bowl",
#     "bell_pepper_on_plate": "plate",
#     "carrot_on_plate": "plate",
#     "grapes_in_bowl": "bowl",
#     "lemon_in_bowl": "bowl",
# }

def build_dataset(data_prefix,
                  suite_name,
                  benchmark_name, 
                  seq_len, 
                  frame_stack,
                  shape_meta,
                  n_demos,
                  hdf5_cache_mode='low_dim',
                  extra_obs_modality=None,
                  obs_seq_len=1, 
                  load_obs=True,
                  load_image=True,
                  load_depth=True,
                  task_embedding_format="clip",
                  load_next_obs=False,
                  stats_mode=False,
                  action_keys=('actions',),
                  ):
    n_tasks = len(benchmarks[benchmark_name])
    task_names = benchmarks[benchmark_name]
    
    manip_datasets = []
    descriptions = []
    if stats_mode:
        obs_modality = {'rgb': [], 'depth': [], 'low_dim': list(shape_meta['observation']['lowdim'].keys())}
    else:
        obs_modality = {
            'rgb': list(shape_meta['observation']['rgb'].keys()) if load_image else [],
            'depth': list(shape_meta['observation']['depth'].keys()) if load_depth else [],
            'low_dim': list(shape_meta['observation']['lowdim'].keys())
        }
    if extra_obs_modality is not None:
        for key in extra_obs_modality:
            obs_modality[key] = obs_modality[key] + extra_obs_modality[key]
    
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    for i in trange(n_tasks):
        task_name = task_names[i]
        task_i_dataset = get_dataset(
            dataset_path=os.path.join(
                data_prefix, suite_name, task_name_to_path(task_name)
            ),
            obs_modality=obs_modality,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            frame_stack=frame_stack,
            load_obs=load_obs,
            n_demos=n_demos,
            hdf5_cache_mode=hdf5_cache_mode,
            load_next_obs=load_next_obs,
            dataset_keys=(),
            action_keys=action_keys,
        )
        task_description = instructions[task_name]
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
    task_embs = get_task_embs(task_embedding_format, descriptions)
    datasets = [
        SequenceVLDataset(ds, task_id=i, **emb) for i, (ds, emb) in enumerate(zip(manip_datasets, task_embs))
    ]
    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    concat_dataset = ConcatDataset(datasets)
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: {benchmark_name}")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")
    return concat_dataset

def get_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    load_obs=True,
    few_demos=None,
    n_demos=None,
    load_next_obs=False,
    dataset_keys=None,
    action_keys=None,
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
        action_keys=action_keys,
        dataset_keys=dataset_keys,
        load_next_obs=load_next_obs,
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


def get_task_embs(task_embedding_format, descriptions):
    logging.set_verbosity_error()
    if task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
        key = 'task_emb'
    elif task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
        key = 'task_emb'
    elif task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
        key = 'task_emb'
    elif task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=25,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
        key = 'task_emb'
    elif task_embedding_format == "lang":
        task_embs = descriptions
        key = 'lang_inst'
    else:
        raise ValueError(f"Unknown task embedding format: {task_embedding_format}")
    return [{key: task_emb} for task_emb in task_embs]
