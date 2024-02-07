# Real-time_musculotendon_kinematics
We developed NN for real-time estimation of musculotedon kinematics.
For a given model (e.g., OpenSim model), MTU lengths, moment arms, and lines of action can be defined as non-linear functions of generalised coordinates (GC) (joint angles). The relationship between GC and MTU kinematics can be approximated using multi-dimensional cubic B-splines (MCBS). However, MCBS are subject specific as they require pre-calculation of MTU kinematics for each subject, as model geometry affects the GC-MTU kinemaitcs relationship.

CREATE ENVIRONMENT
conda env create -f environment.yml

RUN MAIN.py

ADD FURTHER DESCRIPTION WHEN I CAN BE BOTHERED
We developed a nueral network method for estimation of MTU kinematics that considers both GC, and subject geometry.
We start with a base model (e.g., Uhlrich 2022 DOI), and
This project provides data and to reproduce the results found in "ADD DOI":
