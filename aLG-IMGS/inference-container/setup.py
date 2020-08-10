import nltk
from mxnet import gluon

nltk.download('wordnet')
nltk.download('stopwords')
gluon.model_zoo.vision.resnet50_v1(pretrained=True)
