<div id="top" align="center">

<img src=misc/taptap.png width=150 />

Generative Table Pre-training Empowers Models for Tabular Prediction
-----------------------------
<h3> |<a href="https://arxiv.org/abs/2305.09696"> Paper </a> | 
<a href="example.py"> Example </a> |  
<a href="https://huggingface.co/models?search=ztphs980/taptap"> 🤗 Model </a>
</h3>

This is the official Github repository for the paper "Generative Table Pre-training Empowers Models for Tabular Prediction" by Tianping Zhang, Shaowen Wang, Shuicheng Yan, Jian Li, and Qian Liu.

## Overview

Recently, the topic of table pre-training has attracted considerable research interest. However, how to employ table pre-training to boost the performance of tabular prediction (e.g., [Housing price prediction](https://www.coursera.org/projects/tensorflow-beginner-predicting-house-prices-regression)) remains an open challenge. In this project, we present **TapTap**, the first attempt that leverages table pre-training to empower models for tabular prediction.

<img src=misc/taptap_overview.jpg width=750 />

The TapTap model is firstly pre-trained on the pre-training corpus,
and then fine-tuned on the downstream table. During both pre-training and fine-tuning, tables are serialized into sequences via textual encoding, and TapTap is trained to predict them token by token. During inference, TapTap is prompted to sample values for “___” in data prompts, and the filled values build up a synthetic table. Finally, once the backbone model has yielded labels for the synthetic table, it can be used to strengthen the backbone model. In theory TapTap can be applied to any backbone model!

## Code

The fine-tuning code for TapTap is available in this repository. It includes:

- `taptap_trainer.py`: code for fine-tuning TapTap on the downstream dataset
...

## Datasets

### Pre-training Corpus

We have uploaded our pre-training corpus to Huggingface datasets. You can download it from [here](https://huggingface.co/datasets/).

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@article{zhang2023generative,
  title={Generative Table Pre-training Empowers Models for Tabular Prediction},
  author={Zhang, Tianping and Wang, Shaowen and Yan, Shuicheng and Li, Jian and Liu, Qian},
  journal={arXiv preprint arXiv:2305.09696},
  year={2023}
}
```


## Acknowledgement

- [GreaT](https://github.com/kathrinse/be_great): TapTap is inspired a lot by the awesome work of GreaT. We thank the authors of GreaT for releasing their codebase.
- [Huggingface](https://huggingface.co/): We use the Huggingface transformers framework to pre-train / fine-tune our models. We thank the team of Huggingface for their great work.