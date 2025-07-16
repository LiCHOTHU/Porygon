tasks=(square_d1 coffee_d1 threading_d1)
robots=(UR5e Kinova3 IIWA)

for task in ${tasks[@]}; do
    for robot in ${robots[@]}; do
        python quest/utils/generate_mimicgen_robot_change_init_states_simple.py \
            task=mimicgen_hybrid_base task.task_name=${task} +robot=${robot}
    done
done
