# CAT: A Contextualized Conceptualization and Instantiation Framework for Commonsense Reasoning

This is the official code and data repository for the [ACL2023](https://2023.aclweb.org/) (Main Conference) paper:
CAT: A Contextualized Conceptualization and Instantiation Framework for Commonsense Reasoning.

![Overview](demo/overview.png)

## 1. Download Dataset/Model Checkpoints

The AbstractATOMIC dataset, including both annotated part and CAT's pseudo-labeled part, is available
at [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wwangbw_connect_ust_hk/EnA7X6PkeE5Dll9sdlwxuG4BH8zw-Bpdtc5kw3L70Shu5g).

Our finetuned DeBERTa-v3-Large and GPT2 model checkpoints for four tasks are available
at [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wwangbw_connect_ust_hk/EnA7X6PkeE5Dll9sdlwxuG4BH8zw-Bpdtc5kw3L70Shu5g).

## 2. Required Packages

Required packages are listed in `requirements.txt`. Install them by running:

```bash
pip install -r requirements.txt
```

## 3. Training



## 4. Citing this work

Please use the bibtex below for citing our paper:

```bibtex
@inproceedings{CAT,
  author       = {Weiqi Wang and
                  Tianqing Fang and
                  Baixuan Xu and
                  Chun Yi Louis Bo and
                  Yangqiu Song and 
                  Lei Chen},
  title        = {CAT: A Contextualized Conceptualization and Instantiation Framework for Commonsense Reasoning},
  year         = {2023},
  booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, {ACL} 2023}
}
```