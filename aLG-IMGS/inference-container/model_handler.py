"""
ModelHandler defines an example model handler for load and inference requests for MXNet CPU models
"""
from collections import namedtuple
import glob
import json
import logging
import re

import numpy as np
import pandas as pd
import os, time
import autogluon as ag
import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
import pickle
from io import StringIO
from autogluon import TabularPrediction as task
import tensorflow as tf
from tensorflow.keras import layers

import re, string, base64
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from io import StringIO, BytesIO
from html.parser import HTMLParser
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

class ModelHandler(object):
    def __init__(self):
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir") 
        gpu_id = properties.get("gpu_id")
        print('initializing')

        model_path = model_dir+'/'
        model_config = {}
        with open(model_path+'model_config.pkl', 'rb') as f:
            model_config = pickle.load(f)
        print("random value:", model_config['columns'])

        columns = model_config['columns']
        columns_text = model_config['columns_text']
        columns_num = model_config['columns_num']
        columns_cat = model_config['columns_cat']
        columns_img = model_config['columns_img']
        has_text = model_config['has_text']
        has_num = model_config['has_num']
        has_cat = model_config['has_cat']
        has_img = model_config['has_img']
        catmap = model_config['catmap']
        invcatmap = model_config['invcatmap']
        numcats = model_config['numcats']
        cats = model_config['cats']
        wholistic_input_size = model_config['wholistic_input_size']
        target_column = 'label'
        
        def create_model():
            model = tf.keras.Sequential()
            model.add(layers.Dense(
                int(wholistic_input_size*float(2/3))+numcats,
                input_shape=(wholistic_input_size,),
                activation='relu')
            )
            model.add(layers.Dense(int(wholistic_input_size*float(1/3))+numcats, activation='relu'))
            model.add(layers.Dense(numcats, activation='sigmoid'))
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                optimizer='adam', metrics=['accuracy'])
            return model
        model = create_model()
        print('pmodel loaded.')

        # train on each existing modality
        predictor_text = None
        if has_text:
            predictor_text = task.load(model_path+'text/')
            print('done loading text model')
        predictor_num = None
        if has_num:
            predictor_num = task.load(model_path+'num/')
            print('done loading num model')
        predictor_cat = None
        if has_cat:
            predictor_cat = task.load(model_path+'cat/')
            print('done loading cat model')
        predictor_img = None
        if has_img:
            predictor_img = []
            for i in range(len(columns_img)):
                task.load(model_path+'img/'+columns_img[i])
            print('done loading img model')
            
        self.predictor_text = predictor_text
        self.predictor_num = predictor_num
        self.predictor_cat = predictor_cat
        self.predictor_img = predictor_img 

        self.model_config = model_config
        self.model = model
        self.model_path = model_path

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        print('preprocessing')

        self.model.load_weights(self.model_path+'wholistic/wholistic_model')
        print('wholistic model loaded.')
        columns = self.model_config['columns'].copy()
        columns.remove('label')

        raw_data = []
        for idx, data in enumerate(request):
            raw_data.append(data.get('body'))

        data_csvs = [StringIO(str(x, 'utf-8')) for x in raw_data]
        data_pds = [pd.read_csv(x) for x in data_csvs]
        data_pd = pd.concat(data_pds)
        data_pd.columns = columns

        return data_pd

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in NDArray
        """
        print('inferencing')
        import numpy as np
        import pandas as pd
        #from pandarallel import pandarallel
        import sagemaker
        import boto3
        import os, time
        import autogluon as ag
        import mxnet as mx
        from mxnet import nd, gluon, init, autograd
        from mxnet.gluon import nn
        #pandarallel.initialize()

        mx.random.seed(127)
        contexts = [mx.cpu()]
        
        predictor_text = self.predictor_text
        predictor_num = self.predictor_num
        predictor_cat = self.predictor_cat
        predictor_img = self.predictor_img
        model = self.model
        model_config = self.model_config

        columns = model_config['columns']
        columns_text = model_config['columns_text']
        columns_num = model_config['columns_num']
        columns_cat = model_config['columns_cat']
        columns_img = model_config['columns_img']
        has_text = model_config['has_text']
        has_num = model_config['has_num']
        has_cat = model_config['has_cat']
        has_img = model_config['has_img']
        catmap = model_config['catmap']
        invcatmap = model_config['invcatmap']
        numcats = model_config['numcats']
        cats = model_config['cats']
        target_column = 'label'

        mappedlabels = pd.DataFrame()

        test_dataset = model_input
        print('data read')

        def clean_text(data, labelled=False):
            lemmatizer = WordNetLemmatizer()
            nltk.download('wordnet')
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            # scrub possible html
            # stackoverflow.com/questions/753052/strip-html-from-strings-in-python
            def strip_tags(html):
                s = MLStripper()
                s.feed(html)
                return s.get_data()
            # lowercases and removes special characters
            def clean_val(val):
                text = val
                text = strip_tags(text)
                if type(val) == str:
                    text = text.lower().strip()
                    text = re.compile(r'[%s]' % re.escape(string.punctuation)).sub(' ', text)
                    text = re.sub(r'\s+', ' ', text)
                    words = [w for w in text.split(" ") if not w in stop_words]
                    text = " ".join([lemmatizer.lemmatize(w) for w in words])
                return text
            # get text cols
            text_cols = [x for x in data.columns if x.endswith('_text')]
            if len(text_cols) == 0:
                print('no text columns found.')
                return pd.DataFrame(), None
            print('text columns found:', text_cols)
            text_data = data[text_cols].copy()
            # lazy impute text
            text_data = text_data.fillna('')
            # clean text cols
            for col in text_cols:
                print('text cleaning:', col)
                text_data[col] = text_data[col].apply(clean_val)
            text_label = None
            if labelled:
                print('working on labels...')
                text_label = mappedlabels.copy()
                map2bad = text_data[text_cols[0]] == ''
                for col in text_cols[1:]:
                    map2bad = np.logical_and(map2bad, text_data[col] == '')
                text_label[map2bad] = numcats
            print("done transforming text data.")
            return text_data, text_label
        def clean_num(data, labelled=False):
            # get num cols
            num_cols = [x for x in data.columns if x.endswith('_num')]
            if len(num_cols) == 0:
                print('no numeric columns found.')
                return pd.DataFrame(), None
            print('numeric columns found:', num_cols)
            num_data = data[num_cols].copy()
            # impute numeric data
            num_data = num_data.fillna(0)
            num_label = None
            if labelled:
                print('working on labels...')
                num_label = mappedlabels.copy()
            print("done transforming numeric data.")
            return num_data, num_label
        def clean_cat(data, labelled=False):
            # get cat cols
            cat_cols = [x for x in data.columns if x.endswith('_cat')]
            if len(cat_cols) == 0:
                print('no categorical columns found.')
                return pd.DataFrame(), None
            print('categorical columns found:', cat_cols)
            cat_data = data[cat_cols].copy()
            # impute categorical data
            cat_data = cat_data.fillna('unknown')
            cat_data_out = pd.get_dummies(cat_data)
            cat_label = None
            if labelled:
                print('working on labels...')
                cat_label = mappedlabels.copy()
            print("done transforming categorical data.")
            return cat_data_out, cat_label
        def clean_image(data, labelled=False):
            #import cv2
            # normalize image
            def normalizeimg(img):
                img = img.transpose((2, 0, 1)).expand_dims(axis=0)
                rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
                rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
                return (img.astype('float32') / 255 - rgb_mean) / rgb_std
            def cleanimg(img_bytes):
                if img_bytes == '':
                    return normalizeimg(mx.nd.array([[[0]*3]*224]*244))
                img = mx.image.imdecode(base64.b64decode(img_bytes))
                img = mx.image.resize_short(img, 256)
                img, _ = mx.image.center_crop(img, (224, 224))
                return normalizeimg(img)
            # get pretrained resnet50
            def getresnet():
                net = gluon.model_zoo.vision.resnet50_v1(pretrained=True, ctx=contexts)
                print('image pre-model created...')
                return net
            # get cat cols
            img_cols = [x for x in data.columns if x.endswith('_image')]
            if len(img_cols) == 0:
                print('no image columns found.')
                return pd.DataFrame(), None
            print('image columns found:', img_cols)
            img_data = data[img_cols].copy()
            # impute categorical data
            img_data = img_data.fillna('')
            # initialize model to transform images to resnet outputs
            img_model = getresnet()
            # create suggested img labels
            map2bads =  []
            #img_data_out = []
            for col in img_cols:
                map2bads.append(img_data[col] == '')
                img_data[col] = img_data[col].apply(cleanimg).apply(img_model)
                #temp_img_col = pd.DataFrame(img_data[col].apply(cleanimg))
                #print(temp_img_col)
                #img_data_out.append(temp_img_col.apply(img_model, result_type='expand'))
            image_label = []
            if labelled:
                print('working on labels...')
                import functools
                mlabs = mappedlabels.copy()
                for i in range(len(map2bads)):
                    image_label.append(mlabs.copy())
                    image_label[i][map2bads[i]] = numcats
            print("done transforming image data.")
            #return img_data_out, image_label
            return img_data, image_label

        text_data, _ = clean_text(test_dataset)
        num_data, _ = clean_num(test_dataset)
        cat_data, _ = clean_cat(test_dataset)
        img_data, _ = clean_image(test_dataset)


        # ### phase one - per-modality training

        from autogluon import TabularPrediction as task

        if has_text:
            print('loading text data')
            agluon_text_train_data = task.Dataset(text_data)
        if has_num:
            print('loading num data')
            agluon_num_train_data = task.Dataset(num_data)
        if has_cat:
            print('loading cat data')
            agluon_cat_train_data = task.Dataset(cat_data)
        if has_img:
            print('loading img data')
            agluon_img_train_data = []
            for i in range(len(img_data.columns)):
                from itertools import zip_longest
                curr_img_feature = pd.DataFrame.from_records(zip_longest(
                    *img_data.iloc[:, i].apply(lambda x: x[0].asnumpy()).values)).transpose()
                agluon_img_train_data.append(task.Dataset(curr_img_feature))
            #for i in range(len(img_data)):
            #    agluon_img_train_data.append(task.Dataset(img_data[i]))
        print('done')

        if has_text:
            preds_text = predictor_text.predict_proba(agluon_text_train_data)
            print('done generating text unimodal preds')
        if has_num:
            preds_num = predictor_num.predict_proba(agluon_num_train_data)
            print('done generating num unimodal preds')
        if has_cat:
            preds_cat = predictor_cat.predict_proba(agluon_cat_train_data)
            print('done generating cat unimodal preds')
        if has_img:
            preds_img = []
            for i in range(len(predictor_img)):
                preds_img.append(predictor_img[i].predict_proba(agluon_img_train_data[i]))
            print('done generating img unimodal preds')

        # ### phase 2 - wholistic training
        # create data
        wholistic_test = pd.DataFrame()
        if has_text:
            wholistic_test = pd.concat([wholistic_test, pd.DataFrame(preds_text)], axis=1)
        if has_num:
            wholistic_test = pd.concat([wholistic_test, pd.DataFrame(preds_num)], axis=1)
        if has_cat:
            wholistic_test = pd.concat([wholistic_test, pd.DataFrame(preds_cat)], axis=1)
        if has_img:
            wholistic_test = pd.concat([wholistic_test, pd.DataFrame(preds_img)], axis=1)
        wholistic_input_size = wholistic_test.shape[1]

        print('predicting...')
        preds = model.predict(wholistic_test.values)
        print('done')

        def getmaplabs(pred):
            return pred.tolist().index(max(pred))
        def unmap(mappedlab):
            return invcatmap[mappedlab]

        predlabs = pd.DataFrame(preds).apply(getmaplabs, axis=1)
        predlabs = predlabs.apply(unmap)

        submission = test_dataset[['ID']]
        submission['label'] = predlabs
        return submission

    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        print('postprocessing')
        
        return [['ID=%s, label=%d' %(inference_output.iloc[i, 0], inference_output.iloc[i, 1]) for i in range(len(inference_output))]]
        
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        print('INTERNAL handling')
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        print('SERVICE NOT INITIALIZED. INITIALIZING...')
        _service.initialize(context)

    if data is None:
        print('DATA IS NONE')
        return None

    print('SERVICE HANDLE CALLED UWU')
    return _service.handle(data, context)
