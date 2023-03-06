# SSL Baselines for Biosignal Feature Extraction
Implementations of various published works on self-supervised learning approaches to biosignal feature extraction.

## Dataset (Update March, 2023)
The dataset used in this repository has recently been made available upon individual request (please direct requests privately to zac.brown@duke.edu).

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
 - Currently only tested on Windows 10 OS with NVIDIA GPU and Pytorch (therefore any models/pipelines reliant on LSTMs/GRUs have not been fully tested - see https://github.com/pytorch/pytorch/issues/27837)

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
 - PhaseSwap FCN Embedder and Upstream Decoder
 - SeqCLR Embedders (Convolutional and Recurrent) as well as Upstream Decoder (though currently the PhaseSwap FCN Embedder and Upstream Decoder - slightly modified - architectures are being used due to limited computational resources)
#### data_loaders.py
 - data loading function for training RP, TS, CPC, PhaseSwap, and SeqCLR tasks
 - data loading function for downstream task
#### train.py
 - upstream training functions
 - downstream training (with cross-validation)

## To-Do's 
(last updated 02/06/2021)
 - Define/Implement necessary NN models, data preprocessing/loading, upstream training loops, and downstream training loops for TIDNet
