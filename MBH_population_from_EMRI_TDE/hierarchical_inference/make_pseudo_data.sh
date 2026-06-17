#!/bin/bash

GALAXIES=100

python prep_pseudo_data.py \
    --GALAXIES $GALAXIES \
    --POPULATION A

python prep_pseudo_data.py \
    --GALAXIES $GALAXIES \
    --POPULATION B


# MAKE PLOTS 

# python plotting_pseudo_data.py \
#     --GALAXIES $GALAXIES \
#     --POPULATION A

# python plotting_pseudo_data.py \
#     --GALAXIES $GALAXIES \
#     --POPULATION B
        