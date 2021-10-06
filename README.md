# multimodal-algorithm
An algorithm to facilitate the training on and inferencing of multimodal data, along with the necessary machinery needed to package the algorithm up for AWS Sagemaker. The algorithm was built as my intern project at AWS, and I received permission from my manager to display it here on GitHub. For more information - including setup, Sagemaker usage, and implementation details - please see the attached full-length README at the "Multimodal Data Algorithm.pdf" document.

# Use Case
The main use cases of multimodal classification, in the context of consumer goods sellers, are to provide better product suggestions and a better customer experience. Scholastic, for example, has a need to classify books into genres based on the cover, description, and page length. Netflix categorizes movies and TV shows into genres based on similar information. A secondary use case is also the labelling and taking down of “restricted products” - in the case of Amazon, for example, this would include the accurate labelling of items like firearms, to aid in their removal from the market.

## Purpose of the "CONTAINER train" and "CONTAINER inference" Jupyter notebook files

CONTAINER train is meant to give the user a idea of what the algorithm in aLG-IMGS/training-container does.

CONTAINER inference can be used not only to understand how to use the training artifacts provided by sagemaker after the training container is used to inference, but also to manually inference in the case that the model files are too large for the inferencing container to handle during a batch transform job on Sagemaker (still a WIP).

The training container is based off of the AWS mxnet training toolkit image, found here: https://github.com/aws/sagemaker-mxnet-training-toolkit
