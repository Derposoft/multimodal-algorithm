import ast
import argparse
import logging
import warnings
import os
import json
import glob
import subprocess
import sys
import boto3
import pickle
import pandas as pd
from collections import Counter
from timeit import default_timer as timer

sys.path.insert(0, 'package')
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import autogluon as ag
    from autogluon import TabularPrediction as task
    from autogluon.task.tabular_prediction import TabularDataset

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(['du','-sh', path]).split()[0].decode('utf-8')

def __load_input_data(path: str) -> TabularDataset:
    """
    Load training data as dataframe
    :param path:
    :return: DataFrame
    """
    input_data_files = os.listdir(path)
    try:
        input_dfs = [pd.read_csv(f'{path}/{data_file}') for data_file in input_data_files]
        return pd.concat(input_dfs) #task.Dataset(df=pd.concat(input_dfs))
    except:
        print(f'No csv data in {path}!')
        return None

from html.parser import HTMLParser
import re, string, base64
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from io import StringIO, BytesIO
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

def train(args): 
    # ## imports
    import numpy as np
    import pandas as pd
    from pandarallel import pandarallel
    import sagemaker
    import boto3
    import os, time
    import autogluon as ag
    import mxnet as mx
    from mxnet import nd, gluon, init, autograd
    from mxnet.gluon import nn
    import pickle
    import json

    os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
    pandarallel.initialize()

    mx.random.seed(127)
    #gpus = mx.test_utils.list_gpus()
    contexts = [mx.cpu()] #[mx.gpu(i) for i in gpus] if len(gpus) > 0 else [mx.cpu()]

    # ## load data / parameters
    with open('/opt/ml/input/config/hyperparameters.json') as f:
        hyperparameters = json.load(f)

    train_dataset_path = args.train
    model_path = '/opt/ml/model/'
    n_hours = float(args.n_hours) #float(hyperparameters.n_hours)
    n_mins = float(args.n_mins) #float(hyperparameters.n_mins)
    n_secs = float(args.n_secs) #float(hyperparameters.n_secs)
    print("running training for hours, mins, secs=", n_hours, n_mins, n_secs)
    time_limits = int((60 * 60 * n_hours + 60 * n_mins + n_secs) / float(5))
    eval_metric = 'accuracy'
    id_column = 'ID'

    # text/numeric data
    is_distributed = len(args.hosts) > 1
    host_rank = args.hosts.index(args.current_host)
    dist_ip_addrs = args.hosts
    dist_ip_addrs.pop(host_rank)

    # Load training and validation data
    print(f'Train files: {os.listdir(args.train)}')
    ########################### HERE DATASET= LINE COMMENT SWAP
    dataset = __load_input_data(args.train)
    #dataset = pd.read_csv('/usr/local/bin/train_shorter.csv')
    print('data read')

    # create category mapping (to define non-clashing dummy category)
    target_column = "label"
    if target_column not in dataset.columns:
        raise ValueError("Train Dataset must include 'label'!")
    cats = dataset[target_column].unique()
    numcats = len(cats)
    intcats = list(range(numcats))
    catmap = dict(zip(cats, intcats))
    invcatmap = dict(zip(intcats, cats))

    # map labels to new categories
    labels = dataset[target_column]
    mappedlabels = pd.Series([catmap[x] for x in labels])


    # ## data clean / suggested label generation

    def clean_text(data, labelled=False):
        lemmatizer = WordNetLemmatizer()
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
            text_data[col] = text_data[col].parallel_apply(clean_val)
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
                return nd.array([])#normalizeimg(mx.nd.array([[[0]*3]*224]*244))
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
        badimg_pred = img_model(normalizeimg(mx.nd.array([[[0]*3]*224]*244)))
        def get_pred(x):
            if len(x) == 0:
                return badimg_pred
            else:
                return img_model(x)
        map2bads =  []
        for col in img_cols:
            print('cleaning:', col)
            map2bads.append(img_data[col] == '')
            img_data[col] = img_data[col].parallel_apply(cleanimg)
            print('images cleaned. now pre-predicting...')
            img_data[col] = img_data[col].parallel_apply(get_pred)
        image_label = []
        if labelled:
            print('working on labels...')
            import functools
            mlabs = mappedlabels.copy()
            for i in range(len(map2bads)):
                image_label.append(mlabs.copy())
                image_label[i][map2bads[i]] = numcats
        print("done transforming image data.")
        return img_data, image_label

    text_data, text_label = clean_text(dataset, labelled=True)
    num_data, num_label = clean_num(dataset, labelled=True)
    cat_data, cat_label = clean_cat(dataset, labelled=True)
    img_data, img_label = clean_image(dataset, labelled=True)

    # do we have these features?
    has_text = (len(text_data) != 0)
    has_num = (len(num_data) != 0)
    has_cat = (len(cat_data) != 0)
    has_img = (len(img_data) != 0)

    # ## training

    # ### phase one - per-modality training
    from autogluon import TabularPrediction as task

    if has_text:
        print('loading text data')
        agluon_text_train_data = task.Dataset(pd.concat([text_data, pd.Series(text_label, name='label')], axis=1))
    if has_num:
        print('loading num data')
        agluon_num_train_data = task.Dataset(pd.concat([num_data, pd.Series(num_label, name='label')], axis=1))
    if has_cat:
        print('loading cat data')
        agluon_cat_train_data = task.Dataset(pd.concat([cat_data, pd.Series(cat_label, name='label')], axis=1))
    if has_img:
        print('loading img data')
        agluon_img_train_data = []
        for i in range(len(img_data.columns)):
            from itertools import zip_longest
            curr_img_feature = pd.DataFrame.from_records(zip_longest(
                *img_data.iloc[:, i].parallel_apply(lambda x: x[0].asnumpy()).values)).transpose()
            agluon_img_train_data.append(task.Dataset(
                pd.concat([curr_img_feature, pd.Series(img_label[i], name='label')], axis=1)))
    print('done')

    # train on each existing modality
    if has_text:
        predictor_text = task.fit(
            train_data=agluon_text_train_data,
            label=target_column,
            eval_metric=eval_metric,
            presets='optimize_for_deployment',
            time_limits=time_limits,
            id_columns=[id_column],
            #ngpus_per_trial=8,
            nthreads_per_trial=os.cpu_count(),
            verbosity=3,
            problem_type='multiclass',
            hyperparameters=args.quality,
            output_directory=model_path+'text/'
        )
        print('done training text')
    if has_num:
        predictor_num = task.fit(
            train_data=agluon_num_train_data,
            label=target_column,
            eval_metric=eval_metric,
            presets='optimize_for_deployment',
            time_limits=time_limits,
            id_columns=[id_column],
            #ngpus_per_trial=8,
            nthreads_per_trial=os.cpu_count(),
            verbosity=3,
            problem_type='multiclass',
            hyperparameters=args.quality,
            output_directory=model_path+'num/'
        )
        print('done training num')
    if has_cat:
        predictor_cat = task.fit(
            train_data=agluon_cat_train_data,
            label=target_column,
            eval_metric=eval_metric,
            presets='optimize_for_deployment',
            time_limits=time_limits,
            id_columns=[id_column],
            #ngpus_per_trial=8,
            nthreads_per_trial=os.cpu_count(),
            verbosity=3,
            problem_type='multiclass',
            hyperparameters=args.quality,
            output_directory=model_path+'cat/'
        )
        print('done training cat')
    if has_img:
        predictor_img = []
        for i in range(len(agluon_img_train_data)):
            predictor_img.append(task.fit(
                train_data=agluon_img_train_data[i],
                label=target_column,
                eval_metric=eval_metric,
                presets='optimize_for_deployment',
                time_limits=time_limits,
                id_columns=[id_column],
                #ngpus_per_trial=8,
                nthreads_per_trial=os.cpu_count(),
                verbosity=3,
                problem_type='multiclass',
                hyperparameters=args.quality,
                output_directory=model_path+'img/'+img_data.columns[i]
            ))
        print('done training img')

    if has_text:
        preds_text = predictor_text.predict_proba(agluon_text_train_data.drop(columns=['label']))
        print('done generating text unimodal preds')
    if has_num:
        preds_num = predictor_num.predict_proba(agluon_num_train_data.drop(columns=['label']))
        print('done generating num unimodal preds')
    if has_cat:
        preds_cat = predictor_cat.predict_proba(agluon_cat_train_data.drop(columns=['label']))
        print('done generating cat unimodal preds')
    if has_img:
        preds_img = []
        for i in range(len(predictor_img)):
            preds_img.append(predictor_img[i].predict_proba(agluon_img_train_data[i].drop(columns=['label'])))
        print('done generating img unimodal preds')

    # ### phase 2 - wholistic training
    # create data
    wholistic_train = pd.DataFrame()
    if has_text:
        wholistic_train = pd.concat([wholistic_train, pd.DataFrame(preds_text)], axis=1)
    if has_num:
        wholistic_train = pd.concat([wholistic_train, pd.DataFrame(preds_num)], axis=1)
    if has_cat:
        wholistic_train = pd.concat([wholistic_train, pd.DataFrame(preds_cat)], axis=1)
    if has_img:
        for i in range(len(preds_img)):
            wholistic_train = pd.concat([wholistic_train, pd.DataFrame(preds_img[i])], axis=1)
    wholistic_input_size = wholistic_train.shape[1]

    # save model-specific values to pickle file
    model_config = {
        "columns": dataset.columns.tolist(),
        "columns_text": text_data.columns.tolist(),
        "columns_num": num_data.columns.tolist(),
        "columns_cat": cat_data.columns.tolist(),
        "columns_img": img_data.columns.tolist(),
        "has_text": has_text,
        "has_num": has_num,
        "has_cat": has_cat,
        "has_img": has_img,
        "catmap": catmap,
        "invcatmap": invcatmap,
        "numcats": numcats,
        "cats": cats,
        "wholistic_input_size": wholistic_input_size,
        "columns_cat_dummies": cat_data.columns.tolist(),
    }
    with open(model_path+'model_config.pkl', 'wb') as f:
        pickle.dump(model_config, f)
    
    import tensorflow as tf
    from tensorflow.keras import layers

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

    history = model.fit(
        wholistic_train.values,
        mappedlabels.values,
        batch_size=32,
        epochs=int(args.n_epochs),
        validation_split=0.2
    )

    model.save_weights(model_path+'wholistic/wholistic_model')

    # Files summary
    print(f'Model export summary:')
    print(f"/opt/ml/model/: {os.listdir('/opt/ml/model/')}")
    #models_contents = os.listdir('/opt/ml/model/models')
    #print(f"/opt/ml/model/models: {models_contents}")
    print(f"/opt/ml/model directory size: {du('/opt/ml/model/')}\n")

# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type','bool', lambda v: v.lower() in ('yes', 'true', 't', '1'))

    # Environment parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--n_hours', type=str, default=str(0))
    parser.add_argument('--n_mins', type=str, default=str(5))
    parser.add_argument('--n_secs', type=str, default=str(0))
    parser.add_argument('--n_epochs', type=str, default=str(50))
    parser.add_argument('--quality', type=str, default="toy")
    # Arguments to be passed to task.fit()
    parser.add_argument('--fit_args', type=lambda s: ast.literal_eval(s),
                        default="{'presets': ['optimize_for_deployment']}",
                        help='https://autogluon.mxnet.io/api/autogluon.task.html#tabularprediction')
    # Additional options
    parser.add_argument('--feature_importance', type='bool', default=True)

    return parser.parse_args()


if __name__ == "__main__":
    start = timer()
    args = parse_args()
    
    # Convert optional fit call hyperparameters from strings
    if 'hyperparameters' in args.fit_args:
        for model_type,options in args.fit_args['hyperparameters'].items():
            assert isinstance(options, dict)
            for k,v in options.items():
                args.fit_args['hyperparameters'][model_type][k] = eval(v) 
 
    # Print SageMaker args
    print('fit_args:')
    for k,v in args.fit_args.items():
        print(f'{k},  type: {type(v)},  value: {v}')
        
    # Make test data optional
    if os.environ.get('SM_CHANNEL_TESTING'):
        args.test = os.environ['SM_CHANNEL_TESTING']
    else:
        args.test = None

    train(args)

    # Package inference code with model export
    subprocess.call('mkdir /opt/ml/model/code'.split())
    #subprocess.call('cp /opt/ml/code/inference.py /opt/ml/model/code/'.split())
    #subprocess.call('cp columns.pkl /opt/ml/model/code/'.split())

    elapsed_time = round(timer()-start,3)
    print(f'Elapsed time: {elapsed_time} seconds. Training Completed!')
