# SSL Baselines for Biosignal Feature Extraction
Implementations of various published works on self-supervised learning approaches to biosignal feature extraction.

### References:
 - https://github.com/jstranne/mouse_self_supervision
 - Banville et al's arxiv.org/pdf/2007.16104.pdf for RP, TS, and CPC upstream SSL tasks
 - Cheng et al's https://arxiv.org/pdf/2007.04871.pdf for SACL upstream SSL pipeline
 - Mohsenvad et al's http://proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf for SeqCLR upstream SSL pipeline
 - The ThinkerInvariance (TIDNet) SSL pipeline described in https://iopscience.iop.org/article/10.1088/1741-2552/abb7a7/pdf by Demetres Kostas and Frank Rudzicz, and their associated code at https://github.com/SPOClab-ca/ThinkerInvariance
 - PhaseSwap as described in https://arxiv.org/pdf/2009.07664.pdf by Abdelhak Lemkhenter and Paolo Favaro

## Requirements:
 - Pytorch (and dependencies)
 - CUDA

## File Descriptions:

### Important Files:
 - README.md
 - data_utils.py
 - models.py
 - data_loaders.py
 - train.py
 
#### data_utils.py
 - list_training_files
 - extract_data
 - create_windowed_dataset
#### models.py
 - Stager_net_practice
 - Embedders for RP, TS, CPC
 - Downstream classifier model
#### data_loaders.py
 - data loading function for training all three upstream tasks (RP, TS, CPC)
 - data loading function for downstream task
#### train.py
 - upstream training functions
 - downstream training (with cross-validation)

## To-Do's 
(last updated 01/12/2021)
 - Define NN models necessary for implementing SACL, SeqCLR, TIDNet, and PhaseSwap
 - Implement data preprocessing and loading for SACL and SeqCLR
 - Implement data preprocessing and loading for TIDNet and PhaseSwap
 - Implement training loops for all upstream tasks (SACL, SeqCLR, TIDNet, and PhaseSwap)
 - Implement down-stream training loops for all upstream pretrained models (SACL, SeqCLR, TIDNet, and PhaseSwap)
