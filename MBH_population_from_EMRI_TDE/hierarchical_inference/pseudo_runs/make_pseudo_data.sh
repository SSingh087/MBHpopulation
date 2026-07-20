#!/bin/bash

GALAXIES=100000
INCLUDE_LSST=$1

python prep_pseudo_data.py \
    --GALAXIES $GALAXIES \
    --POPULATION A \
    --INCLUDE_LSST $INCLUDE_LSST

python prep_pseudo_data.py \
    --GALAXIES $GALAXIES \
    --POPULATION B \
    --INCLUDE_LSST $INCLUDE_LSST

# # MAKE PLOTS 

python plotting_pseudo_data.py \
    --GALAXIES $GALAXIES \
    --POPULATION A

python plotting_pseudo_data.py \
    --GALAXIES $GALAXIES \
    --POPULATION B
        