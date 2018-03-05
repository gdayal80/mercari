import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from functools import partial
from contextlib import contextmanager
from sklearn.metrics import mean_squared_log_error

import time
import keras as ks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras import backend as K
from multiprocessing.pool import ThreadPool
from sklearn.model_selection import KFold
import tensorflow as tf

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    t1 = time.time()
    print('[{0}] done in {1:.0f} s'.format(name, t1 - t0))

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    return (np.sqrt(mean_squared_log_error(y, y_pred)))

def load_train_data():
    print("Loading training data...")
    
    train = pd.read_table("../input/train.tsv")
    train = train[train['price'] > 0].reset_index(drop=True)
    
    print(train.shape)
    
    return(train)

def load_test_data():
    print("Loading testing data...")
    
    test = pd.read_table("../input/test.tsv")
    #test = pd.read_table("../input/test_stg2.tsv")
    
    print(test.shape)
    
    return(test)


def handle_missing(dataset):
    print("Handling missing values...")
    
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    
    return (dataset)

def do_label_encoding(train, test):
    print("Handling categorical variables...")
    le = LabelEncoder()
    
    le.fit(np.hstack([train.category_name, test.category_name]))
    train.category_name = le.transform(train.category_name)
    test.category_name = le.transform(test.category_name)
    
    le.fit(np.hstack([train.brand_name, test.brand_name]))
    train.brand_name = le.transform(train.brand_name)
    test.brand_name = le.transform(test.brand_name)
    del le
    return([train, test])

def convert_text_to_seq(train, test):
    print("Text to seq process...")
    raw_text = np.hstack([train.item_description.str.lower(), train.name.str.lower()])

    print("   Fitting tokenizer...")
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)
    print("   Transforming text to seq...")

    train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
    test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
    train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
    test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
    
    print("Process Complete...")
    return([train, test])

def add_name_and_item_desc_seq(train, test):
    max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))), np.max(test.seq_name.apply(lambda x: len(x)))])
    max_seq_item_description = np.max([np.max(train.seq_item_description.apply(lambda x: len(x)))
                                   , np.max(test.seq_item_description.apply(lambda x: len(x)))])
    print("max name seq "+str(max_name_seq))
    print("max item desc seq "+str(max_seq_item_description))

    train.seq_name.apply(lambda x: len(x)).hist()
    train.seq_item_description.apply(lambda x: len(x)).hist()
    
    return([train, test])

def cal_max_constants(train, test):
    MAX_NAME_SEQ = 10
    MAX_ITEM_DESC_SEQ = 75
    MAX_TEXT = np.max([np.max(train.seq_name.max())
                   , np.max(test.seq_name.max())
                  , np.max(train.seq_item_description.max())
                  , np.max(test.seq_item_description.max())])+2
    MAX_CATEGORY = np.max([train.category_name.max(), test.category_name.max()])+1
    MAX_BRAND = np.max([train.brand_name.max(), test.brand_name.max()])+1
    MAX_CONDITION = np.max([train.item_condition_id.max(), test.item_condition_id.max()])+1
    
    return([MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ, MAX_TEXT, MAX_CATEGORY, MAX_BRAND, MAX_CONDITION])

def get_target_scaler():
    return(MinMaxScaler(feature_range=(-1, 1)))

def transform_price(train, target_scaler):
    train["target"] = np.log1p(train['price'].values.reshape(-1, 1))
    train["target"] = target_scaler.fit_transform(train.target.values.reshape(-1,1))
    
    return(train)

def split_train(train):
    cv = KFold(n_splits=20, shuffle=True, random_state=42)
    dtrain_ids, dvalid_ids = next(cv.split(train))
    dtrain, dvalid = train.iloc[dtrain_ids], train.iloc[dvalid_ids]
    print(dtrain.shape)
    print(dvalid.shape)
    
    return([dtrain, dvalid])

def get_keras_data(dataset, MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ):
    X = {
         'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        ,'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ)
        ,'brand_name': np.array(dataset.brand_name)
        ,'category_name': np.array(dataset.category_name)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
    }
    return X

def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def get_model(X_train, MAX_TEXT, MAX_CATEGORY, MAX_BRAND, MAX_CONDITION):
    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    #Embeddings layers
    emb_name = Embedding(MAX_TEXT, 50)(name)
    emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    
    #rnn layer
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)
    
    #main layer
    model_in = concatenate([
         Flatten() (emb_brand_name)
        ,Flatten() (emb_category_name)
        ,Flatten() (emb_item_condition)
        ,rnn_layer1
        ,rnn_layer2
        ,num_vars
    ])
    
    #output
    out = Dense(128, activation="relu") (model_in)
    out = Dense(64, activation="relu") (out)
    out = Dense(64, activation="relu") (out)
    out = Dense(1) (out)
    
    #model
    model = Model([name, item_desc, brand_name
                   , category_name, item_condition, num_vars], out)
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3), metrics=["mae", rmsle_cust])
    
    return model

def fit_predict(xs, target_scaler, const):
    MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ, MAX_TEXT, MAX_CATEGORY, MAX_BRAND, MAX_CONDITION = const
    X_train, X_valid, dtrain, dvalid, X_test, process = xs
    epochs = 1
    
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict on {0}'.format(process)):
        K.set_session(sess)
        model = get_model(X_train, MAX_TEXT, MAX_CATEGORY, MAX_BRAND, MAX_CONDITION)
        print('Starting model.fit on {0}.'.format(process))
        for i in range(3):
            with timer('epoch {0} on {1}'.format(i + 1, process)):
                model.fit(X_train, dtrain.target, epochs=epochs, batch_size=2**(11+i)
                          , validation_data=(X_valid, dvalid.target), verbose=0)
        
        print('Starting model.predict on {0} on labelled sample.'.format(process))
        val_preds = model.predict(X_valid)
        val_preds = target_scaler.inverse_transform(val_preds.reshape(-1, 1))[:, 0]
        val_preds = np.expm1(val_preds)
                
        #mean_absolute_error, mean_squared_log_error
        y_true = np.array(dvalid.price.values)
        y_pred = val_preds
        v_rmsle = rmsle(y_true, y_pred)
        print("RMSLE error on {0}: {1} ".format(process, str(v_rmsle)))
            
        return(model.predict(X_test))

def save_submission(submission, target_scaler, preds):
    preds = target_scaler.inverse_transform(preds.reshape(-1, 1))[:, 0]
    preds = np.expm1(preds)
    
    submission["price"] = np.array(preds)
    
    submission.to_csv("sample_submission_stg2.csv", index=False)
    
    print("Submission Saved")

def main():
    train = load_train_data()
    test = load_test_data()
    
    train = handle_missing(train)
    test = handle_missing(test)
    
    print(train.shape)
    print(test.shape)
    
    target_scaller = get_target_scaler()
    
    train, test = do_label_encoding(train, test)
    train, test = convert_text_to_seq(train, test)
    train, test = add_name_and_item_desc_seq(train, test)
    
    train = transform_price(train, target_scaller)
    
    constants = cal_max_constants(train, test)
    MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ, MAX_TEXT, MAX_CATEGORY, MAX_BRAND, MAX_CONDITION = constants
    
    dtrain, dvalid = split_train(train)
    
    X_train = get_keras_data(dtrain, MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ)
    X_valid = get_keras_data(dvalid, MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ)
    X_test = get_keras_data(test, MAX_NAME_SEQ, MAX_ITEM_DESC_SEQ)
    
    submission = test[["test_id"]]
    
    del train
    del test
    
    print("Next few steps might take long time, so relax.")
    
    with ThreadPool(processes=4) as pool:
        xs = [[X_train, X_valid, dtrain, dvalid, X_test, "Thread 1"], 
              [X_train, X_valid, dtrain, dvalid, X_test, "Thread 2"],
              [X_train, X_valid, dtrain, dvalid, X_test, "Thread 3"],
              [X_train, X_valid, dtrain, dvalid, X_test, "Thread 4"]]
        y_pred = np.mean(pool.map(partial(fit_predict, target_scaler=target_scaller, const=constants), xs), axis=0)
        save_submission(submission, target_scaller, y_pred)

if __name__ == '__main__':
    main()