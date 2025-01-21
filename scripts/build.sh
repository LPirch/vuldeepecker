#!/bin/bash

image_id=$(docker images docker-vuldeepecker:latest -q --no-trunc)

TAR_TMP_FILE=/tmp/docker-vuldeepecker-$(id -u).tar
echo "[+] Saving $image_id to $TAR_TMP_FILE"
docker save $image_id  -o $TAR_TMP_FILE
echo "[+] Building Apptainer container"
apptainer build vuldeepecker.sif docker-archive:$TAR_TMP_FILE
echo "[+] Cleanup of docker tar image"
rm -f $TAR_TMP_FILE
echo "[+] All done!"
