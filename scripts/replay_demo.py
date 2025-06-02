import os
import h5py
import hydra
import numpy as np
from tqdm import trange
from scipy.spatial.transform import Rotation
from hydra.utils import instantiate
from libero import benchmark

def process_demo(old_demo, env):
    """
    Replay demo in simulation to collect observations with custom image dimensions.
    
    Args:
        old_demo (dict): Original demo data
        env: Environment instance with custom image dimensions
    """
    actions = old_demo['actions']
    states = old_demo['states']
    init_state = states[0]

    obs, info = env.reset(init_state)

    new_demo = {
        'actions': [],
        'abs_actions': [],
    }
    for key in obs:
        new_demo[key] = []

    for t in trange(len(actions), disable=True):
        for key in obs:
            new_demo[key].append(obs[key])

        obs, reward, terminated, truncated, info = env.step(actions[t])

        # read pos and ori from robots
        controller = env.env.robots[0].controller
        goal_pos = controller.goal_pos
        goal_ori = Rotation.from_matrix(controller.goal_ori).as_rotvec()
        abs_action = np.concatenate((goal_pos, goal_ori, actions[t][..., -1:]))
        
        new_demo['actions'].append(actions[t])
        new_demo['abs_actions'].append(abs_action)

    # Convert lists to numpy arrays
    for key in new_demo:
        new_demo[key] = np.array(new_demo[key])
    
    return new_demo

def save_demo(demo, output_path):
    """
    Save the processed demo to a new HDF5 file.
    
    Args:
        demo (dict): Processed demo dictionary
        output_path (str): Path to save the new HDF5 file
    """
    with h5py.File(output_path, 'w') as f:
        group_data = f.create_group('data')
        group = group_data.create_group('demo_0')
        
        # Save all data
        for key, value in demo.items():
            if key == 'states':
                group.create_dataset(key, data=value)
            else:
                group.create_dataset(f'obs/{key}', data=value)
        
        # Set attributes
        group.attrs['num_samples'] = len(demo['actions'])
        group_data.attrs['total'] = len(demo['actions'])

@hydra.main(config_path="../config", 
            config_name='collect_data', 
            version_base=None)
def main(cfg):
    """
    Main function to process a single demo file.
    
    Configuration options:
        input_h5_path (str): Path to the input HDF5 file
        output_h5_path (str, optional): Path to save the output HDF5 file. If not provided,
            will save in the same directory as input with '_resized' suffix.
        task_no (int): Task number for environment instantiation
        img_height (int): Target image height (default: 384)
        img_width (int): Target image width (default: 512)
    """
    # Get input path
    input_path = cfg.input_h5_path
    
    # Get output path
    if hasattr(cfg, 'output_h5_path'):
        output_path = cfg.output_h5_path
    else:
        output_dir = os.path.dirname(input_path)
        output_filename = os.path.basename(input_path).replace('.hdf5', '_resized.hdf5')
        output_path = os.path.join(output_dir, output_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the demo
    with h5py.File(input_path, 'r') as f:
        demo = f['data']['demo_0']
    
    # Instantiate environment with custom image dimensions
    env_factory = instantiate(cfg.task.env_factory)

    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[cfg.task.benchmark_name]()

    num_tasks = benchmark_instance.get_num_tasks()
    task_files = [os.path.join(source_dir, benchmark_instance.get_task_demonstration(i).split('/')[1]) for i in range(num_tasks)]
    task_names = benchmark_instance.get_task_names()

    env = env_factory(
        benchmark=benchmark_instance,
        task_id=task_names[0],
        img_height=cfg.img_height if hasattr(cfg, 'img_height') else 384,
        img_width=cfg.img_width if hasattr(cfg, 'img_width') else 512
    )
    
    # Process the demo
    print(f"Processing demo from {input_path}")
    new_demo = process_demo(demo, env)
    
    # Save the processed demo
    print(f"Saving processed demo to {output_path}")
    save_demo(new_demo, output_path)
    print("Done!")

if __name__ == "__main__":
    main() 