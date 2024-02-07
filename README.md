# Real-time_musculotendon_kinematics
We developed NN for real-time estimation of musculotedon kinematics.
For a given model (e.g., OpenSim model), MTU lengths, moment arms, and lines of action can be defined as non-linear functions of generalised coordinates (GC) (joint angles). The relationship between GC and MTU kinematics can be approximated using multi-dimensional cubic B-splines (MCBS). However, MCBS are subject specific as they require pre-calculation of MTU kinematics for each subject, as model geometry affects the GC-MTU kinemaitcs relationship.


# Create Environment
conda env create -f environment.yml

# Evaluate MTU-NN models
Run MAIN.py

# Add furtehr description when I can be bothered

