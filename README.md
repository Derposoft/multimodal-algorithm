# multimodal-algorithm
A Sagemaker algorithm to facilitate the training on and inferencing of multimodal data.

For more - including setup, usage, and implementation information - please see the attached full-length README at the "Multimodal Data Algorithm.pdf" document.


## Purpose of CONTAINER train and CONTAINER inference

CONTAINER train is meant to give the user a idea of what the algorithm in aLG-IMGS/training-container does.

CONTAINER inference can be used not only to understand how to use the training artifacts provided by sagemaker after the training container is used to inference, but also to manually inference in the case that the model files are too large for the inferencing container to handle during a batch transform job on Sagemaker (still a WIP).

The training container is based off of the AWS mxnet training toolkit image, found here: https://github.com/aws/sagemaker-mxnet-training-toolkit
