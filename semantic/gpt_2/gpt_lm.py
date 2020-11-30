import os
import math
import numpy as np
from keras_gpt_2 import load_trained_model_from_checkpoint,get_bpe_from_files

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir,"models/774M")

config_path = os.path.join(model_dir,"hparams.json")
checkpoint_path = os.path.join(model_dir,"model.ckpt")
encoder_path = os.path.join(model_dir,"encoder.json")
vocab_path = os.path.join(model_dir,"vocab.bpe")

print('Load model from checkpoint...')
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
print('Load BPE from files...')
bpe = get_bpe_from_files(encoder_path, vocab_path)

def gpt2_ppl(texts):
    sentences = []
    start_token = bpe.token_dict["<|endoftext|>"]
    for text in texts:
        encode_text = [start_token]
        # for w in text.split():
        #     encode_text.extend(bpe.encode(w))
        encode_text.extend(bpe.encode(text))
        # print(encode_text)
        # res = []
        # for index in encode_text:
        #     res.append(bpe.token_dict_inv[index])
        # print(res)
        y = model.predict(np.array(encode_text))
        pro = 1.0
        ps = []
        tokens = []
        for i,index in enumerate(encode_text[1:]):
            ps.append(float(y[i][0][index]))
            pro *= float(y[i][0][index])
            tokens.append({"token":bpe.token_dict_inv[index],"prob":float(y[i][0][index])})
        ppl = -math.log(pro)/len(ps)
        sentence = {
            "tokens":tokens,
            "ppl":ppl
        }
        sentences.append(sentence)
    return sentences

if __name__ == '__main__':
    texts = ["beautiful flower","successful summaries","there is a book on the desk"]
    res = gpt2_ppl(texts)
    print(res)

            