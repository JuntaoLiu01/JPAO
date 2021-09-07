### classifier

### Steps:

- step 1: download BERT base model from [here](https://github.com/google-research/bert), and put it into `semantic/bert/models`directory.

- step 2: update the model name according to your BERT version in `run_classify_prob.py` :

  ```python
  BERT_BASE_DIR = os.path.join(BASE_DIR,"semantic/bert/models/uncased_L-12_H-768_A-12")
  config_path = os.path.join(BERT_BASE_DIR, "bert_config.json")
  checkpoint_path = os.path.join(BERT_BASE_DIR,"bert_model.ckpt")
  dict_path = os.path.join(BERT_BASE_DIR,"vocab.txt")
  ```

-  运行 `run_classify_prob.py` 

  ```shell
  python run_classify_prob.py
  ```
