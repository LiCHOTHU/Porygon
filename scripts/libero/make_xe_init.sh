# robots=("UR5e" "Kinova3")
robots=("UR5e" "Kinova3" "Sawyer")
late_starts=(0 15 30 45 60 75)

for robot in ${robots[@]}; do
    for start in ${late_starts[@]}; do
        sbatch slurm/run_rtx6000.sbatch python quest/libero_custom/make_robot_change_init.py \
            task=libero_90_hybrid \
            +robot=${robot} \
            +late_start=${start}
    done
done




python quest/libero_custom/make_robot_change_init.py task=libero_90_hybrid +robot=IIWA +late_start=0 & \
python quest/libero_custom/make_robot_change_init.py task=libero_90_hybrid +robot=IIWA +late_start=10 & \
python quest/libero_custom/make_robot_change_init.py task=libero_90_hybrid +robot=IIWA +late_start=20 & \
python quest/libero_custom/make_robot_change_init.py task=libero_90_hybrid +robot=IIWA +late_start=30 & \
python quest/libero_custom/make_robot_change_init.py task=libero_90_hybrid +robot=IIWA +late_start=40 & \
python quest/libero_custom/make_robot_change_init.py task=libero_90_hybrid +robot=IIWA +late_start=50 & \
python quest/libero_custom/make_robot_change_init.py task=libero_90_hybrid +robot=IIWA +late_start=60 & \
python quest/libero_custom/make_robot_change_init.py task=libero_90_hybrid +robot=IIWA +late_start=70 & \
python quest/libero_custom/make_robot_change_init.py task=libero_90_hybrid +robot=IIWA +late_start=80 &
