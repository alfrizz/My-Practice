#!/usr/bin/env bash

while true; do
  unison \
    /home/alfrizz/my_practice \
    "/mnt/g/My Drive/Ingegneria/Data Science GD/My-Practice" \
    -batch \
    -auto \
    -repeat 2 \
    -logfile "$HOME/sync-practice.log" \
    -ignore "Name .ipynb_checkpoints" \
    -ignore "Path *.log"

  sleep 2
done
