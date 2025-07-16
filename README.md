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
```bash
# Install main dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

If you want to run point cloud stuff you need to also install DGL
```bash
uv pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

You'll probably need CLIP for something
```bash
uv pip install git+https://github.com/openai/CLIP.git
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

There are also lots of example scripts in the `scripts` folder to take a look at

## Evaluation

All of the parameters to build policies are stored in the saved checkpoints so you don't need to specify them at eval time. All you need to do is specify the task and point to the checkpoint path. Example to evaluate a mimicgen coffee_d1 policy

```bash
uv run evaluate.py \
    task=mimicgen \
    task.task_name=coffee_d1 \
    checkpoint_path=[path]
```
to export videos instead, replace `evaluate.py` with `export_videos.py`.

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

