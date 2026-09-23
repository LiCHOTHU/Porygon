# Benchmark task demonstrations

Watch [the nine-second montage](benchmark_task_montage.mp4), or use the [footage player](index.html) for the complete clips.

| Clip | Task | Source |
|---|---|---|
| [LIBERO](libero_task_demo.mp4) | LIBERO-90 task 65: put the red mug on the left plate | Local recorded human demonstration, `demo_0`; replayed simulator states |
| [robomimic](robomimic_task_demo.mp4) | Square nut assembly | Official proficient-human demonstration dataset, `demo_0`; replayed simulator states |
| [DMC](dmc_task_demo.mp4) | Cartpole balance, dense reward | Actual simulator with a scripted LQR balancing controller |

These clips show what the tasks look like. They are **not CAST policy evaluations**. Source type is labeled on screen. All clips play at normal speed; the montage selects three seconds from each task. The manipulation replays retain the recorded positions and velocities, hide collision meshes and diagnostic markers, and use the original agent-view camera pose with a widescreen aspect ratio.

The manipulation clips are approximately 6.37 seconds each; DMC is 12 seconds. Videos are H.264, 1280 × 720, 30 fps. The original manipulation state rate is 20 Hz; output frames are duplicated as needed to preserve time. Detailed paths, source hashes, software versions, replay settings, and the DMC controller are recorded in `source/*_provenance.json`.

## Sources

- LIBERO states: `/home/licho/workspace/imitation/data/libero/libero_90_unprocessed/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5`. Task assets come from the existing `/home/licho/workspace/LIBERO` checkout.
- robomimic: [official dataset documentation](https://robomimic.github.io/docs/v0.2/datasets/robomimic_v0.1.html); downloaded `square/ph/low_dim.hdf5` from the linked Stanford dataset host into `/tmp/icra-video-raw/square_ph_low_dim.hdf5`.
- DMC: [official Control Suite implementation](https://github.com/google-deepmind/dm_control). DMC was installed in `/tmp/icra-video-dmc`; existing Conda environments were not modified.

## Reproduce

Run from the repository root. The script accepts `--dataset` and `--demo` overrides for the manipulation recordings. Use `ICRA_LIBERO_ASSETS` to override the local asset directory.

```bash
PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl \
  /home/licho/anaconda3/envs/liberocag/bin/python \
  icra_video/scripts/record_environment_demos.py libero

PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl \
  /home/licho/anaconda3/envs/liberocag/bin/python \
  icra_video/scripts/record_environment_demos.py robomimic

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/icra-video-dmc MUJOCO_GL=egl \
  /home/licho/anaconda3/envs/river/bin/python \
  icra_video/scripts/record_environment_demos.py dmc

python icra_video/scripts/build_environment_montage.py
```

The montage builder also produces `A05_environments_and_results.mp4`: nine seconds of task context followed by 21 seconds of the existing measured ablation animation. Incoming simulation evaluation videos can replace the illustrative clips; the storyboard retains 24 seconds of unused budget for that footage.
