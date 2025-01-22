#!/bin/bash

set -eu

seed=$SLURM_ARRAY_TASK_ID

sif_file="/home/$USER/vuldeepecker.sif"
volume_dir="/home/$USER/data/vuldeepecker"
volumes="raw results"

bind_locations=(
    "/home/$USER/vuldeepecker/docker:/vuldeepecker"
)
for volume in ${volumes}; do
    bind_locations+=( "${volume_dir}/${volume}:/vuldeepecker/data/${volume}" )
done

bind_args=()
for location in ${bind_locations[@]}; do
    bind_args+=("--bind ${location}")
done

container_cmd="cd /vuldeepecker && python -m venv .venv &&source .venv/bin/activate && pip install -r requirements.txt && ./run_experiment.sh ${seed}"

pushd docker > /dev/null
    apptainer run ${bind_args[@]} --nv --writable-tmpfs $sif_file "$@" bash -c "${container_cmd}"
popd > /dev/null
