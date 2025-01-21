#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

seed=$1

sif_file="/home/$USER/vuldeepecker.sif"
volume_dir="/home/$USER/cheetah/data"
volumes="vuldeepecker-in vuldeepecker-out"

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

container_cmd="cd /vuldeepecker && source .venv/bin/activate && ./run_experiment.sh ${seed}"

pushd docker > /dev/null
    apptainer run ${bind_args[@]} --nv --writable-tmpfs $sif_file "$@" bash -c "${container_cmd}"
popd > /dev/null
