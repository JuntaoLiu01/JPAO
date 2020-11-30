import os
import pandas as pd
import numpy  as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
set_session(sess)
 
from keras.layers import Input,Lambda,Dense,Layer
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
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

token_dict = {}
with open(dict_path,"r") as rf:
    for line in rf:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)
        
def load_data(mode="train"):
    data = []
    with open(os.path.join(DATA_DIR,"{}.csv".format(mode)),"r") as rf:
        for i,line in enumerate(rf):
            if i == 0:
                continue
            adj,noun,label = line.strip().split(",")
            data.append((adj+" " +noun,int(label)))
    return data

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
            X1,X2,Y = [],[],[]
            for i in ids:
                d = self.data[i]
                text = d[0]
                x1,x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == ids[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y  = seq_padding(Y)
                    yield [X1,X2],Y
                    [X1,X2,Y] = [],[],[]
 
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

def build_dense_model():
    bert_model = load_trained_model_from_checkpoint(config_path,checkpoint_path,seq_len=None)
    for l in bert_model.layers:
        l.trainable = False

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in,x2_in])
    x = Lambda(lambda x:x[:,0])(x)
    x = Dense(64,activation="relu",name="dense")(x)
    p = Dense(1,activation="sigmoid",name="ouput")(x)
    model = Model([x1_in,x2_in],p)
    model.compile(loss="binary_crossentropy",optimizer=Adam(5e-5),metrics=["accuracy"])
    model.summary()
    return model

def build_bert_model():
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x = BertLayer(
                config_path,
                checkpoint_path,
                open_layers=[
                    "Encoder-12-MultiHeadSelfAttention_Wq",
                    # "Encoder-12-FeedForward_W2",
            ])([x1_in,x2_in])
    p = Dense(1,activation="sigmoid")(x)
    model = Model([x1_in,x2_in],p)
    model.compile(loss="binary_crossentropy",optimizer=Adam(5e-5),metrics=["accuracy"])
    model.summary()
    # print("trainable weights:")
    # for x in model.trainable_weights:
    #     print(x.name,x.shape)
    return model

def train_model(model,train_data,valid_data,plot=True):
    save_best = ModelCheckpoint(
        "./model/encoder12Wq_model_{epoch:02d}_{val_loss:.2f}_{val_acc:.2f}.h5",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        period=1
    )
    callbacks = [save_best]
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

def train_from_base_model(m_fn,train_data,valid_data,model_type="bert"):
    if model_type == "bert":
        model = build_bert_model()
    else:
        model = build_dense_model()
    model.load_weights(m_fn)
    save_best = ModelCheckpoint(
        "./model/dense_model_{epoch:02d}_{val_loss:.2f}_{val_acc:.2f}.h5",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        period=1
    )
    callbacks = [save_best]
    train_D = data_generater(train_data)
    valid_D = data_generater(valid_data)
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=EPOCHS,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=callbacks
    )

def predict(test_data,model_fn,thresh=0.5,model_type="bert"):
    if model_type == "bert":
        model = build_bert_model()
    else:
        model = build_dense_model()
    model.load_weights(model_fn)
    test_D = data_generater(test_data,shuffle=False)
    test_pred = model.predict_generator(test_D.__iter__(),steps=len(test_D))
    count_1 = 0
    count_0 = 0
    for x in test_pred:
        if x >= 0.5:
            count_1 += 1
        else:
            count_0 += 1
    print(count_1)
    print(count_0)
    # wf = open("./data/test_result.csv","w")
    # wf.write("prob,label\n")
    # for x in test_pred:
    #     x = x[0]
    #     if x >= 0.5:
    #         wf.write(str(x)+",1\n")
    #     else:
    #         wf.write(str(x)+",0\n")

def plot_results(history,acc=True,loss=True):
    print(history.history.keys())
    if acc:
        plt.plot(history.history["acc"])
        plt.plot(history.history["val_acc"])
        plt.legend(["train","valid"],loc="upper left")
        plt.title("Model Accuracy");plt.xlabel("epoch");plt.ylabel("accuracy")
        plt.savefig("./data/imgs/acc.pdf")
        plt.close()
    if loss:
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.legend(["train","valid"],loc="upper left")
        plt.title("Model Loss");plt.xlabel("epoch");plt.ylabel("loss")
        plt.savefig("./data/imgs/loss.pdf")
        plt.close()

def cal_p_r_f1():
    test_data = load_data("valid")
    model = build_bert_model()
    model.load_weights("./model/encoder12Wq.h5")

    test_D = data_generater(test_data,shuffle=False)
    test_pred = model.predict_generator(test_D.__iter__(),steps=len(test_D))

    y_true = [x[1] for x in test_data]
    y_pred = []
    for x in test_pred:
        x = x[0]
        if x >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    
    p_count = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                p_count.append(1)
            else:
                p_count.append(0)
    p = sum(p_count)/len(p_count)

    r_count = []
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                r_count.append(1)
            else:
                r_count.append(0)
    r = sum(r_count)/len(r_count)
    print("precision:{}".format(str(p)))
    print("recall:{}".format(str(r)))
    print("f1-value:{}".format(str(2*r*p/(p+r))))

if __name__ == '__main__':
    train_data = load_data("train")
    valid_data = load_data("valid")

    model = build_bert_model()
    train_model(model,train_data,valid_data)

    # test_data = load_data("valid")
    # predict(test_data,"./model/encoder12Wq.h5")

    # cal_p_r_f1()