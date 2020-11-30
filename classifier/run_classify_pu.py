import os
import copy
import random
import pandas as pd
import numpy  as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
set_session(sess)
 
from keras.layers import Input,Lambda,Dense,Layer,Concatenate
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,Callback
from keras_bert import load_trained_model_from_checkpoint,Tokenizer
from keras_lr_multiplier import LRMultiplier
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR,"classifier/data")

BERT_BASE_DIR = os.path.join(BASE_DIR,"semantic/bert/models/uncased_L-12_H-768_A-12")
config_path = os.path.join(BERT_BASE_DIR, "bert_config.json")
checkpoint_path = os.path.join(BERT_BASE_DIR,"bert_model.ckpt")
dict_path = os.path.join(BERT_BASE_DIR,"vocab.txt")

MODEL_DIR = "./model"
EPOCHS = 100
PI = 0.7
BETA = 0.2
YITA = 0.5

token_dict = {}
with open(dict_path,"r") as rf:
    for line in rf:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)
        
def load_data(mode="train_prob"):
    pos_data = []
    neg_data = []
    with open(os.path.join(DATA_DIR,"{}.csv".format(mode)),"r") as rf:
        for i,line in enumerate(rf):
            if i == 0:
                continue
            adj,noun,label,pan,pna,rep = line.strip().split(",")
            if label == "1":
                pos_data.append([adj+" " +noun,int(label),int(label),float(pan),float(pna),float(rep)])
            else:
                neg_data.append([adj+" " +noun,int(label),int(label),float(pan),float(pna),float(rep)])
    data = []
    for d in pos_data[:int(len(pos_data) * 0.6)]:
        data.append(d)
    print("pos data:%d" % len(data))
    for d in pos_data[int(len(pos_data) * 0.6):] + neg_data:
        d[1] = -1
        data.append(d)
    print("unlabeled data:%d" % (len(data) - int(len(pos_data) * 0.6)))
    random.shuffle(data)
    true_data = copy.deepcopy(data)
    for i,d in enumerate(data):
        del data[i][2]
        del true_data[i][1]
    # print(data[:10])
    # print(true_data[:10])
    return data,true_data

def seq_padding(X,padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x,[padding] * (ML-len(x))]) if len(x) < ML else x for x in X
    ])

class data_generater:
    def __init__(self,data,batch_size=32,shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data)//self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.shuffle = shuffle
    
    def __len__(self,):
        return self.steps
    
    def __iter__(self,):
        while True:
            ids = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(ids)
            X1,X2,X3,Y = [],[],[],[]
            for i in ids:
                d = self.data[i]
                text = d[0]
                x1,x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                X3.append(d[2:])
                Y.append([y])
                if len(X1) == self.batch_size or i == ids[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    X3 = seq_padding(X3)
                    Y  = seq_padding(Y)
                    yield [X1,X2,X3],Y
                    [X1,X2,X3,Y] = [],[],[],[]
 
class BertLayer(Layer):
    def __init__(self,config_path,checkpoint_path,open_layers,seq_len=None,**kwargs):
        super(BertLayer,self).__init__(**kwargs)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.seq_len = seq_len
        self.open_layers = open_layers
        self.output_size = 768
    
    def build(self,input_shape):
        self.bert = load_trained_model_from_checkpoint(self.config_path,self.checkpoint_path,seq_len=self.seq_len)
        for w in self.bert.weights:
            # print(w.name,w.shape)
            for l_name in self.open_layers:
                if w.name.find(l_name) != -1:
                    self._trainable_weights.append(w)
                    break
        for w in self.bert.weights:
            if not w in self._trainable_weights:
                self._non_trainable_weights.append(w)   
        super(BertLayer,self).build(input_shape)

    def call(self,inputs):
        x = self.bert(inputs)
        outputs = x[:,0]
        return outputs

    def compute_output_shape(self,input_shape):
        return tuple([input_shape[0][0],self.output_size])

class ACCCallback(Callback):
    def __init__(self,true_train_data,true_valid_data,thresh=0.0,**kwargs):
        super(ACCCallback, self).__init__(**kwargs)
        self.train_D = data_generater(true_train_data,shuffle=False)
        self.train_y = [[x[1]] for x in true_train_data]
        self.valid_D = data_generater(true_valid_data,shuffle=False)
        self.valid_y = [[x[1]] for x in true_valid_data]
        self.thresh = thresh
        self.max_acc = 0.0

    def on_train_begin(self, logs=None):
        return
    
    def on_train_end(self, logs=None):
        print("max acc on valid data:{}".format(str(self.max_acc)))
        return
    
    def on_epoch_begin(self, epoch, logs=None):
        return
    
    def on_epoch_end(self, epoch, logs=None):
        train_acc = self.cal_acc(self.train_D,self.train_y)
        print("train acc:{}".format(str(train_acc)))
        valid_acc = self.cal_acc(self.valid_D,self.valid_y)
        print("valid acc:{}".format(str(valid_acc)))
        if valid_acc > self.max_acc:
            if os.path.exists("./model/pu_model_{}.h5".format(str(round(self.max_acc,2)))):
                os.remove("./model/pu_model_{}.h5".format(str(round(self.max_acc,2))))
            self.max_acc = valid_acc
            self.model.save_weights("./model/pu_model_{}.h5".format(str(round(self.max_acc,2))))
        return

    def cal_acc(self,data,y_true):
        y_pred = self.model.predict_generator(data.__iter__(),steps=len(data))
        y_pred = np.where(y_pred >= self.thresh,1,0)
        acc = np.mean(np.equal(y_true,y_pred))
        return acc

    def on_batch_begin(self, batch, logs=None):
        return
    
    def on_batch_end(self, batch, logs=None):
        return

def double_hinge_loss(y_true,y_pred):
    g = tf.multiply(y_true,y_pred)
    l = 0.5 - 0.5 * g
    zero = tf.zeros_like(l)
    l = tf.where(l < 0.0,zero,l)
    g_neg = -1 * g
    r = tf.where(l < g_neg,g_neg,l)
    r = tf.reduce_mean(r)
    return r

def pu_double_hinge_loss(y_true,y_pred):
    pos_index = tf.where(tf.equal(y_true,-1))[:,0]
    unl_index = tf.where(tf.equal(y_true,1))[:,0]
    pos_y_true = tf.gather(y_true,pos_index)
    pos_y_pred = tf.gather(y_pred,pos_index)

    unl_y_true = tf.gather(y_true,unl_index)
    unl_y_pred = tf.gather(y_pred,unl_index)

    j_neg = double_hinge_loss(unl_y_true,unl_y_pred) - PI * double_hinge_loss(pos_y_true,-1 * pos_y_pred)
    j_pos = PI * double_hinge_loss(pos_y_true,pos_y_pred)
    
    beta = BETA * tf.ones_like(j_neg)
    loss = tf.cond(j_neg < -1 * beta,lambda:-1 * YITA * j_neg,lambda:tf.add(j_pos,j_neg))
    return loss

def pu_true_loss(y_true,y_pred):
    pos_index = tf.where(tf.equal(y_true,-1))[:,0]
    unl_index = tf.where(tf.equal(y_true,1))[:,0]
    pos_y_true = tf.gather(y_true,pos_index)
    pos_y_pred = tf.gather(y_pred,pos_index)

    unl_y_true = tf.gather(y_true,unl_index)
    unl_y_pred = tf.gather(y_pred,unl_index)

    j_neg = double_hinge_loss(unl_y_true,unl_y_pred) - PI * double_hinge_loss(pos_y_true,-1 * pos_y_pred)
    j_pos = PI * double_hinge_loss(pos_y_true,pos_y_pred)

    beta = BETA * tf.ones_like(j_neg)
    loss = tf.cond(j_neg < -1 * beta,lambda:beta,lambda:tf.add(j_pos,j_neg))
    return loss
    
def build_bert_model():
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(3,))
    bert_x = BertLayer(
                config_path,
                checkpoint_path,
                open_layers=[
                    "Encoder-12-MultiHeadSelfAttention_Wq",
            ])([x1_in,x2_in])
    bert_x = Dense(2,activation="elu")(bert_x)
    prob_x = Dense(2,activation="elu",input_shape=(3,))(x3_in)
    x = Concatenate(axis=-1)([bert_x,prob_x])
    p = Dense(1,activation="tanh")(x)
    model = Model([x1_in,x2_in,x3_in],p)
    model.compile(loss=pu_double_hinge_loss,optimizer=Adam(5e-4),metrics=[pu_true_loss])
    model.summary()
    return model

def train_model(model,train_data,valid_data,true_train,true_valid,plot=True):
    # save_best = ModelCheckpoint(
    #     "./model/prob_encoder12Wq_model_{epoch:02d}_{val_loss:.2f}_{val_acc:.2f}.h5",
    #     monitor="val_loss",
    #     mode="min",
    #     save_best_only=True,
    #     save_weights_only=True,
    #     period=1
    # )
    callbacks = [ACCCallback(true_train,true_valid,thresh=0.0)]
    train_D = data_generater(train_data)
    valid_D = data_generater(valid_data)
    history = model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=EPOCHS,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=callbacks
    )
    if plot:
        plot_results(history)

def predict(test_data,model_fn,thresh=0.5):
    model = build_bert_model()
    model.load_weights(model_fn)
    test_D = data_generater(test_data,shuffle=False)
    test_pred = model.predict_generator(test_D.__iter__(),steps=len(test_D))
    wf = open("./data/test_result.csv","w")
    wf.write("prob,label\n")
    for x in test_pred:
        x = x[0]
        if x >= 0.5:
            wf.write(str(x)+",1\n")
        else:
            wf.write(str(x)+",0\n")

def plot_results(history,loss=True):
    print(history.history.keys())
    if loss:
        plt.plot(history.history["pu_true_loss"])
        plt.plot(history.history["val_pu_true_loss"])
        plt.legend(["train","valid"],loc="upper left")
        plt.title("Model Loss");plt.xlabel("epoch");plt.ylabel("loss")
        plt.savefig("./data/imgs/pu_loss.pdf")
        plt.close()

if __name__ == '__main__':
    train_data,true_train_data = load_data("train_prob")
    valid_data,true_valid_data = load_data("valid_prob")

    model = build_bert_model()
    train_model(model,train_data,valid_data,true_train_data,true_valid_data,plot=True)

    # test_data = load_data("test_prob")
    # predict(test_data,"./model/encoder12Wq.h5")