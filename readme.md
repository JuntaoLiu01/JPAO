## (adjective, concept)-CSK codes and data

### File Description

- **crawl_data.py:** get Google-Sytactic-Ngrams automatically.
- **clean_data.py:** filter module.
- **cluster_data.py:** clustering module.
- **concept_data.py:** concept generation.

### Steps

- Python packages
  - tensorflow-1.14.0
  - bert4keras-0.9.1
  - Keras-2.3.1
  - For other missing packages, download by `pip install XXX`

- Download [Google-N-grams corpus](http://commondatastorage.googleapis.com/books/syntactic-ngrams/index.html)

  ```shell
  python crawl_data.py
  ```

  - proxy for linking to Google if you can not connect Google server

- Data preprocessing and filtering module

  ```shell
  python clean_data.py
  ```

  - or you can download our processed file: https://drive.google.com/file/d/1-VTO-_rO70IRgwvBfbHzIa5YKxuHC6pp/view?usp=sharing
  
- Clustering module

  ```shell
  python cluster_data.py
  ```

- Conceptualization and  evaluation module

  ```shell
  python concept_data.py
  ```

  - before doing so, you should：
    - collect concept tree from [Microsoft Concept Graph](https://concept.research.microsoft.com/Home/Introduction)， see `mdl` directory for details.
    - download BERT model from [here](https://github.com/google-research/bert) ， and put it at `semantic/bert/models` directory.


### Data Resources

- **Processed Data:** https://drive.google.com/file/d/10PscHZct60fWZDYnZLReLdBNzrq3Wh8J/view?usp=sharing
- **(adjective, concept)-CSK:** https://drive.google.com/file/d/1irBGSYU_o6S9RePvOMb5UWc5hrKZw1sm/view?usp=sharing


### Information

- **Liu juntao:** jtliu19@fudan.edu.cn

