> [!WARNING]
> **This repository is deprecated and no longer maintained.**
>
> Development has moved to **[PRECISE-research/cornish-mtu-kinematics](https://github.com/PRECISE-research/cornish-mtu-kinematics)**, a pip-installable PyTorch package with pre-trained models for several OpenSim models, OpenSim `.mot`/`.sto` interoperability, and a real-time C++ kernel. Please use that repository for new work and for issues.
>
> The original code below is kept for reference only.

---

# Real-time_musculotendon_kinematics
We developed NN for real-time estimation of musculotedon kinematics.
For a given model (e.g., OpenSim model), MTU lengths, moment arms, and lines of action can be defined as non-linear functions of generalised coordinates (GC) (joint angles). The relationship between GC and MTU kinematics can be approximated using multi-dimensional cubic B-splines (MCBS). However, MCBS are subject specific as they require pre-calculation of MTU kinematics for each subject, as model geometry affects the GC-MTU kinemaitcs relationship.
Our NN use subject-specific dimensions (OpenSim scale factors) ind addition to GC to estimate MTU kinematics.
The developed NN provide real-time MTU-kinematics, with similar accuracy to MCBS, and result in negligible ~1% differences in downstream calculation of joint contact forces


# Create Environment
conda env create -f environment.yml

# Evaluate MTU-NN model prediction for musculotendon lengths and moment arms
Run main.py

# Future updates
Tensorflow Lite conversion and evalutaion of computation time.
Additional OpenSim (base) models.

