# Imitation

This repository is intended to be a jumping off point for imitation learning projects

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Follow these steps to set up the development environment:

### 1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository:
```bash
git clone https://github.com/yourusername/imitation.git
cd imitation
```

### 3. Create and sync a virtual environment:
```bash
uv sync
```

### 4. Install dependencies:

If you want to run point cloud stuff you need to also install DGL according to the instructions [here](https://www.dgl.ai/pages/start.html). On my system I used the following:
```bash
uv pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

### 5. (optional) Install LIBERO
First download it
```bash
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd imitation
```
Next, since LIBERO is old we need to manually add the pyproject.toml
```bash
cp imitation/envs/libero/pyproject.toml ../LIBERO/
```
Finally, install it
```bash
uv pip install -e ../LIBERO
```

### 6. (optional) Install MimicGen
First download it
```bash
cd ..
git clone https://github.com/NVlabs/mimicgen.git
cd imitation
```
Next, since MimicGen is old we need to manually add the pyproject.toml
```bash
cp imitation/envs/mimicgen/pyproject.toml ../mimicgen/
```
Finally, install it
```bash
uv pip install -e ../mimicgen
```

### 7. (optional) Install DexMimicGen
This is a bit more involved since DexMimicGen runs on a different version of robosuite and there is no way to square the circle and make everything compatible. I would recommend setting this up in a separate conda environment.
First download it
```bash
cd ..
git clone https://github.com/NVlabs/dexmimicgen.git
cd imitation
```
Next go into the dexmimicgen repo and comment out all the lines in `requirements.txt`. __Do not skip this step!__

Now create a conda environment, activate it and install this package
```bash
conda create -n imitation-dmg python=3.10 -y
conda activate imitation-dmg
pip install -e .
```
I've found it works best if you install robosuite, robosuite_models, dexmimicgen and robomimic from source.
```bash
cd ..
git clone -b v1.5.1 https://github.com/ARISE-Initiative/robosuite.git
pip install -e robosuite/
git clone https://github.com/ARISE-Initiative/robosuite_models.git
pip install -e robosuite_models/
pip install -e dexmimicgen/
TODO: robomimic install instruction
```
Make sure you have the right version of mink
```bash
pip install mink==0.0.10
```

Sometimes you need to do this
```bash
pip install qpsolvers[quadprog]
```

## Hydra

This repository _heavily_ uses Hydra and will be difficult to follow without a working understanding of Hydra. If you're not up to date make sure to read the [Hydra documentation](https://hydra.cc/docs/intro/).

## Data Download / Processing

### LIBERO

First download the dataset. Note that our code base assumes the data is stored in a folder titled `data/` within the Adapt3r repository. If you would like to store it somewhere else please create a symlink. 

Download the data into `data/` by running
```bash
uv run scripts/download_libero.py
```
Note that this file renames the data such that, for LIBERO-90, it is stored in `data/libero/libero_90_unprocessed`

Then, process it into a format that is suitable for our training by running 
```bash
uv run scripts/process_libero_data.py  task=libero_90_data
```
You can minimally modify these instructions to support whichever other LIBERO benchmark you'd like to evaluate on.

### MimicGen

Download the MimicGen data based on the instructions [here](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html). Make sure it ends up in the folder `data/mimicgen/core`.

Next, you'll need to process it to add absolute actions, depth and calibration information. To do this, run the following command (updating the task name to correspond to whichever task you are interested in processing):
```bash
uv run scripts/process_mimicgen.py --hdf5_path data/mimicgen/core/task_dx.hdf5 --output_dir data/mimicgen/core_depth/task_dx.hdf5 --depth
```
Note: This will take a few hours :(.

## Training

Example script to train a diffusion policy with a ResNet backbone on the LIBERO-90 benchmark

```bash
uv run train.py \
    --config-name=train.yaml \
    task=libero \
    algo=diffusion_policy \
    algo/encoder=rgb  \
    algo.chunk_size=8
```

There are also lots of example scripts in the `scripts` folder to take a look at.

To train another policy, replace `algo` with the desired policy (options are `act` and `baku`). To train with DP3 or iDP3, replace `algo/encoder` with `dp3` or `idp3` respectively. To train RGB or RGBD, replace `algo/encoder` with `default` and change `task` to `libero_90_rgb` or `libero_90_rgbd` respectively.

`exp_name` and `variant_name` are used to organize your training runs. In particular, they determind where they are saved. Generally in our workflow, `exp_name` would refer to an experiment encompassing several runs while `variant_name` would be one configuration of parameters encompassing several seeds. For example, if you were sweeping over chunk sizes you might choose `exp_name=chunk_size_sweep` and launch several runs with `chunk_size=${chunk_size} variant_name=chunk_size_${chunk_size}`. Then in WandB, filter by `exp_name` and group by `variant_name`

For debugging, we recommend replacing `--config-name=train.yaml` with `--config-name=train_debug.yaml`, which will change several parameters to make debugging more convenient (enabling TQDM, disabling WandB logging, etc.). 

Other useful training tips:
 - Our pipeline automatically saves a checkpoint after each epoch that is overwritten after the subsequent epoch. If training crashes, you can resume training by simply rerunning the original command (assuming you are not using the debug mode, which creates a unique directory for each run).
 - You may want to adjust the dataloader configuration to better fit your system.
 - Our pipeline, by default, saves a checkpoint which is not overwritten every 10 epochs. You can change this behavior in the configs


## Evaluation

All of the parameters to build policies are stored in the saved checkpoints so you don't need to specify them at eval time. All you need to do is specify the task and point to the checkpoint path. Example to evaluate a mimicgen coffee_d1 policy

```bash
uv run evaluate.py \
    task=mimicgen \
    task.task_name=coffee_d1 \
    checkpoint_path=[path]
```
to export videos instead, replace `evaluate.py` with `export_videos.py`.

To run evaluation with an unseen embodiment, add `task.robot=ROBOT`, where `ROBOT` can be one of {`UR5e`, `Kinova3`, `IIWA`}. To run with an unseen viewpoint, add `task.cam_shift=SIZE` where `SIZE` is one of {`small`, `medium`, `large`} or an angle in radians. Our code base also has old infrastructure for adding distractor objects and randomizing the lighting and colors of objects in the scene, but that is untested and unsupported.


Note: you can override parameters saved in the model (maybe temporal aggregation) using the following
```bash
uv run evaluate.py \
    task=mimicgen \
    task.task_name=coffee_d1 \
    +overrides.temporal_agg=false \
    checkpoint_path=[path]
```

You can change the robot by doing `task.robot=X` and move the camera by doing `task.cam_shift=theta`.

## Design Notes

In this section I ramble about design decisions I've made to help make the repository more readable. Feel free to add if there are things that I've missed that would be helpful to understand

### Hydra Use

I am generally pretty religious with my use of hydra.utils.instantiate. This gives the nice property that most of the objects in the code base are recreatable purely based on python dictionaries, which leads to some nice properties. For example:
 - All parameters passed into any object are visible in the WandB logs. 
 - For evaluation, we can just point to the checkpoint path (where the parameter dictionary is saved) and evaluate without needing to pass in any other information about the policy.

I think that with an understanding of Hydra this makes it much nicer to work with this codebase. However, this paradigm is different from standard Python practices, so I'd suggest giving the [Hydra docs](https://hydra.cc/docs/intro/) a close read before trying to make any substantial changes to this repo.


### Modular policies

Each policy is composed of an observation encoder and an action decoder. The observation encoder processes observations into perception and proprioception tokens and specifies how many of each it will output and what dimension they will be. The action decoder should be designed to condition on this information to condition on arbitrary sequences of encoding tokens from the encoders. It should be the case that any algorithm (ACT, DP, etc) can use any observation encoder.

### Action Representations

The goal is to have maximally flexible action representations without the need to reprocess the dataset for each one. To that end, we do the following with actions:
1. In the shape meta we specify a list of action keys and their corresponding dimensions. Note: in the current version of the repo we assume that for unimanual policies the order is pos, rot, gripper and for bimanual policies the order is r_pos, r_rot, l_pos, l_rot, r_gripper, l_gripper. 
2. The policy has a preprocess actions function that preprocesses actions from the dataset into the format that the network expects. Note that we run this function before computing normalization statistics. The means that by changing the action preprocessing you can implement arbitrary transformations for the actions and the normalization statistics will be computed correctly.
3. There are two postprocessing functions because certain steps in action postprocessing may need to happen before or after temporal aggregation. For example, transforming absolute actions in the end effector coordinate frame back to the world frame would have to happen before action aggregation so that when we aggregate all the actions we are aggregating over are in the same coordinate frame. However we don't want to aggregate across absolute axis angle rotations due to their discontinuities (prefer 6D rotations) so we need to postprocess again after aggregation to transform to the right action representation. These are examples that are implemented here but hopefully there are other settings where this design decision is useful.

## TODOs

- [ ] Better logging
- [ ] Rework HDF5 loading to be one unified set of code across LIBERO, mimicgen and dexmimicgen.
- [ ] Zarr datasets


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

