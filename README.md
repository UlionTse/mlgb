<p align="center">
  <img src="https://github.com/UlionTse/mlgb/blob/main/docs/mlgb_logo.png" width="200"/>
</p>
<p align="center">
  <a href="https://pypi.org/project/mlgb"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/mlgb.svg?color=blue"></a>
  <a href="https://anaconda.org/conda-forge/mlgb"><img alt="Conda - Version" src="https://img.shields.io/conda/vn/conda-forge/mlgb.svg?color=blue"></a>
  <a href="https://pypi.org/project/mlgb"><img alt="PyPI - License" src="https://img.shields.io/pypi/l/mlgb.svg?color=brightgreen"></a>
  <a href="https://pypi.org/project/mlgb"><img alt="PyPI - Python" src="https://img.shields.io/pypi/pyversions/mlgb.svg?color=blue"></a>
  <a href="https://pypi.org/project/mlgb"><img alt="PyPI - Status" src="https://img.shields.io/pypi/status/mlgb.svg?color=brightgreen"></a>
  <a href="https://pypi.org/project/mlgb"><img alt="PyPI - Wheel" src="https://img.shields.io/badge/wheel-yes-brightgreen.svg"></a>
  <a href="https://pypi.org/project/mlgb"><img alt="PyPI - Downloads" src="https://static.pepy.tech/personalized-badge/mlgb?period=total&units=international_system&left_text=downloads&left_color=grey&right_color=blue"></a>
  <a href="https://pypi.org/project/mlgb"><img alt="PyPI - TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.10+-yellow.svg"></a>
  <a href="https://pypi.org/project/mlgb"><img alt="PyPI - PyTorch" src="https://img.shields.io/badge/PyTorch-2.1+-tomato.svg"></a>
</p>

* * *

**MLGB** means **M**achine **L**earning of the **G**reat **B**oss, and is called **「妙计包」**.  
**MLGB** is a library that includes many models of CTR Prediction & Recommender System by TensorFlow & PyTorch.

- [Advantages](#advantages)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Code Examples](#code-examples)
- [Citation](#citation)

## Advantages

- **Easy!** Use `mlgb.get_model(model_name, **kwargs)` to get a complex model.
- **Fast!** Better performance through better code.
- **Enjoyable!** 50+ ranking & matching models to use, 2 languages(TensorFlow & PyTorch) to deploy.

## Supported Models

| ID  | Model Name    | Paper Link                                                                                                                                                                             | Paper Team                                                                   | Paper Year |
| --- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------- |
| <tr><th colspan=5 align="center">:open_file_folder: **Ranking-Model::Normal** :point_down:</th></tr> |
| 1   | LR            | [Predicting Clicks: Estimating the Click-Through Rate for New Ads](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/predictingclicks.pdf)                           | Microsoft                                                                    | 2007       |
| 2   | PLM/MLR       | [Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/pdf/1704.05194.pdf)                                                                | Alibaba                                                                      | 2017       |
| 3   | MLP/DNN       | [Neural Networks for Pattern Recognition](http://diyhpl.us/~bryan/papers2/ai/ahuman-pdf-only/neural-networks/2005-Pattern%20Recognition.pdf)                                           | Christopher M. Bishop(Microsoft, 1997-Present), Foreword by Geoffrey Hinton. | 1995       |
| 4   | DLRM          | [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091.pdf)                                                              | Facebook(Meta)                                                               | 2019       |
| 5   | MaskNet       | [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/pdf/2102.07619.pdf)                                                 | Weibo(Sina)                                                                  | 2021       |
|     |               |                                                                                                                                                                                        |                                                                              |            |
| 6   | DCM/DeepCross | [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)                                        | Microsoft                                                                    | 2016       |
| 7   | DCN           | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/pdf/2008.13535.pdf), [v1](https://arxiv.org/pdf/1708.05123.pdf) | Google(Alphabet)                                                             | 2017, 2020 |
| 8   | EDCN          | [Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf)            | Huawei                                                                       | 2021       |
|     |               |                                                                                                                                                                                        |                                                                              |            |
| 9   | FM            | [Factorization Machines](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf)                                                                                       | Steffen Rendle(Google, 2013-Present)                                         | 2010       |
| 10  | FFM           | [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)                                                                             | NTU                                                                          | 2016       |
| 11  | HOFM          | [Higher-Order Factorization Machines](https://arxiv.org/pdf/1607.07195v2.pdf)                                                                                                          | NTT                                                                          | 2016       |
| 12  | FwFM          | [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)                                                 | Junwei Pan(Yahoo), etc.                                                      | 2018, 2020 |
| 13  | FmFM          | [FM^2: Field-matrixed Factorization Machines for Recommender Systems](https://arxiv.org/pdf/2102.12994v2.pdf)                                                                          | Yahoo                                                                        | 2021       |
| 14  | FEFM          | [FIELD-EMBEDDED FACTORIZATION MACHINES FOR CLICK-THROUGH RATE PREDICTION](https://arxiv.org/pdf/2009.09931v2.pdf)                                                                      | Harshit Pande(Adobe)                                                         | 2020, 2021 |
| 15  | AFM           | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)                                         | ZJU&NUS(Jun Xiao(ZJU), Xiangnan He(NUS), etc.)                               | 2017       |
| 16  | LFM           | [Learning Feature Interactions with Lorentzian Factorization Machine](https://arxiv.org/pdf/1911.09821.pdf)                                                                            | EBay                                                                         | 2019       |
| 17  | IFM           | [An Input-aware Factorization Machine for Sparse Prediction](https://www.ijcai.org/proceedings/2019/0203.pdf)                                                                          | THU                                                                          | 2019       |
| 18  | DIFM          | [A Dual Input-aware Factorization Machine for CTR Prediction](https://www.ijcai.org/proceedings/2020/0434.pdf)                                                                         | THU                                                                          | 2020       |
|     |               |                                                                                                                                                                                        |                                                                              |            |
| 19  | FNN           | [Deep Learning over Multi-field Categorical Data – A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                                                     | UCL(Weinan Zhang(UCL, SJTU), etc.)                                           | 2016       |
| 20  | PNN           | [Product-based Neural Networks for User Response](https://arxiv.org/pdf/1611.00144.pdf)                                                                                                | SJTU&UCL(Yanru Qu(SJTU), Weinan Zhang(SJTU, UCL), etc.)                      | 2016       |
| 21  | PIN           | [Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data](https://arxiv.org/pdf/1807.00311.pdf)                                                   | Huawei(Yanru Qu(Huawei(2017.3-2018.3), SJTU), Weinan Zhang(SJTU, UCL), etc.) | 2018       |
| 22  | ONN/NFFM      | [Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                                                   | NJU                                                                          | 2019       |
| 23  | AFN           | [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://arxiv.org/pdf/1909.03276v2.pdf)                                                                 | SJTU                                                                         | 2019, 2020 |
|     |               |                                                                                                                                                                                        |                                                                              |            |
| 24  | NFM           | [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                                                                  | NUS(Xiangnan He(NUS))                                                        | 2017       |
| 25  | WDL           | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                                                   | Google(Alphabet)                                                             | 2016       |
| 26  | DeepFM        | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)                                                                        | Huawei                                                                       | 2017       |
| 27  | DeepFEFM      | [FIELD-EMBEDDED FACTORIZATION MACHINES FOR CLICK-THROUGH RATE PREDICTION](https://arxiv.org/pdf/2009.09931v2.pdf)                                                                      | Harshit Pande(Adobe)                                                         | 2020, 2021 |
| 28  | FLEN          | [FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/pdf/1911.04690v4.pdf)                                                                                           | Meitu                                                                        | 2019, 2020 |
|     |               |                                                                                                                                                                                        |                                                                              |            |
| 29  | CCPM          | [A Convolutional Click Prediction Model](http://wnzhang.net/share/rtb-papers/cnn-ctr.pdf)                                                                                              | CASIA                                                                        | 2015       |
| 30  | FGCNN         | [Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1904.04447.pdf)                                                           | Huawei                                                                       | 2019       |
| 31  | XDeepFM       | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170v3.pdf)                                                        | Microsoft(Jianxun Lian(USTC, Microsoft(2018.7-Present)), etc.)               | 2018       |
| 32  | FiBiNet       | [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)                                       | Weibo(Sina)                                                                  | 2019       |
| 33  | AutoInt       | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921v2.pdf)                                                           | PKU                                                                          | 2018, 2019 |
| <tr><th colspan=5 align="center">:open_file_folder: **Ranking-Model::Sequential** :point_down:</th></tr> |
| 34  | GRU4Rec       | [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/pdf/1511.06939.pdf)                                                                                   | Telefonica                                                                   | 2015, 2016 |
| 35  | Caser         | [Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding](https://arxiv.org/pdf/1809.07426.pdf)                                                              | SFU                                                                          | 2018       |
| 36  | SASRec        | [Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf)                                                                                                       | UCSD                                                                         | 2018       |
| 37  | BERT4Rec      | [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/pdf/1904.06690.pdf)                                                | Alibaba                                                                      | 2019       |
| 38  | BST           | [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1905.06874.pdf)                                                                         | Alibaba                                                                      | 2019       |
| 39  | DIN           | [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978v4.pdf), [v1](https://arxiv.org/pdf/1706.06978v1.pdf)                                        | Alibaba                                                                      | 2017, 2018 |
| 40  | DIEN          | [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                                                              | Alibaba                                                                      | 2018       |
| 41  | DSIN          | [Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.06482.pdf)                                                                                | Alibaba                                                                      | 2019       |
| <tr><th colspan=5 align="center">:open_file_folder: **Ranking-Model::Multitask** :point_down:</th></tr> |
| 42  | SharedBottom  | [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098.pdf)                                                                                     | Sebastian Ruder(InsightCentre)                                               | 2017       |
| 43  | ESMM          | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/pdf/1804.07931.pdf)                                                 | Alibaba                                                                      | 2018       |
| 44  | MMoE          | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)                                            | Google(Alphabet)                                                             | 2018       |
| 45  | PLE           | [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://www.sci-hub.se/10.1145/3383313.3412236)                       | Tencent                                                                      | 2020       |
| 46  | PEPNet        | [PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information](https://arxiv.org/pdf/2302.01115.pdf)                                          | Kuaishou                                                                     | 2023       |
| <tr><th colspan=5 align="center">:open_file_folder: **Matching-Model** :point_down:</th></tr> |
| 47  | NCF           | [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)                                                                                                                 | NUS(Xiangnan He(NUS), etc)                                                   | 2017       |
| 48  | MatchFM       | [Factorization Machines](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/Rendle2010FM.pdf)                                                                                       | Steffen Rendle(Google, 2013-Present)                                         | 2010       |
| 49  | DSSM          | [Learning deep structured semantic models for web search using clickthrough data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)   | Microsoft                                                                    | 2013       |
| 50  | EBR           | [Embedding-based Retrieval in Facebook Search](https://browse.arxiv.org/pdf/2006.11632.pdf)                                                                                            | Facebook(Meta)                                                               | 2020       |
| 51  | YoutubeDNN    | [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)                                       | Google(Alphabet)                                                             | 2016       |
| 52  | MIND          | [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030.pdf)                                                                        | Alibaba                                                                      | 2019       |
|     |               |                                                                                                                                                                                        |                                                                              |            |

## Installation

```sh
# PYPI
pip install --upgrade mlgb

# Conda
conda install conda-forge::mlgb
```

## Getting Started

```python
import mlgb

# parameters of get_model:
help(mlgb.get_model)

"""
get_model(feature_names, model_name='LR', task='binary', aim='ranking', lang='TensorFlow', device=None, seed=None, **kwargs)
    :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
    :param model_name: str, default 'LR'. Union[`mlgb.ranking_models`, `mlgb.matching_models`, `mlgb.mtl_models`]
    :param task: str, default 'binary'. Union['binary', 'regression', 'multiclass:{int}']
    :param aim: str, default 'ranking'. Union['ranking', 'matching', 'mtl']
    :param lang: str, default 'TensorFlow'. Union['TensorFlow', 'PyTorch', 'tf', 'torch']
    :param device: Optional[str, int], default None. Only for PyTorch.
    :param seed: Optional[int], default None.
    :param **kwargs: more model parameters by `mlgb.get_model_help(model_name)`.
"""

# parameters of model:
mlgb.get_model_help(model_name='LR', lang='tf')

"""
 class LR(tf.keras.src.models.model.Model)
 |  LR(feature_names, task='binary', seed=None, inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False, embed_dim=32, embed_2d_dim=None, embed_l2=0.0, embed_initializer=None, pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_l2=0.0, pool_mv_initializer=None, pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_l2=0.0, pool_seq_initializer=None, linear_if_bias=True, linear_l1=0.0, linear_l2=0.0, linear_initializer=None)
 |  
 |  Methods defined here:
 |  
 |  __init__(self, feature_names, task='binary', seed=None, inputs_if_multivalued=False, inputs_if_sequential=False, inputs_if_embed_dense=False, embed_dim=32, embed_2d_dim=None, embed_l2=0.0, embed_initializer=None, pool_mv_mode='Pooling:average', pool_mv_axis=2, pool_mv_l2=0.0, pool_mv_initializer=None, pool_seq_mode='Pooling:average', pool_seq_axis=1, pool_seq_l2=0.0, pool_seq_initializer=None, linear_if_bias=True, linear_l1=0.0, linear_l2=0.0, linear_initializer=None)
 |      Model Name: LR(LinearOrLogisticRegression)
 |      Paper Team: Microsoft
 |      Paper Year: 2007
 |      Paper Name: <Predicting Clicks: Estimating the Click-Through Rate for New Ads>
 |      Paper Link: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/predictingclicks.pdf
 |      
 |      Task Inputs Parameters:
 |          :param feature_names: tuple(tuple(dict)), must. Embedding need vocabulary size and custom embed_dim of features.
 |          :param task: str, default 'binary'. Union['binary', 'regression']
 |          :param seed: Optional[int], default None.
 |          :param inputs_if_multivalued: bool, default False.
 |          :param inputs_if_sequential: bool, default False.
 |          :param inputs_if_embed_dense: bool, default False.
 |          :param embed_dim: int, default 32.
 |          :param embed_2d_dim: Optional[int], default None. When None, each field has own embed_dim by feature_names.
 |          :param embed_l2: float, default 0.0.
 |          :param embed_initializer: Optional[str], default None. When None, activation judge first, xavier_normal end.
 |          :param pool_mv_mode: str, default 'Pooling:average'. Pooling mode of multivalued inputs. Union[
 |                              'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
 |          :param pool_mv_axis: int, default 2. Pooling axis of multivalued inputs.
 |          :param pool_mv_l2: float, default 0.0. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
 |          :param pool_mv_initializer: Optional[str], default None. When None, activation judge first,
 |                              xavier_normal end. When pool_mv_mode is in ('Weighted', 'Attention'), it works.
 |          :param pool_seq_mode: str, default 'Pooling:average'. Pooling mode of sequential inputs. Union[
 |                              'Attention', 'Weighted', 'Pooling:max', 'Pooling:average', 'Pooling:sum']
 |          :param pool_seq_axis: int, default 1. Pooling axis of sequential inputs.
 |          :param pool_seq_l2: float, default 0.0. When pool_seq_mode is in ('Weighted', 'Attention'), it works.
 |          :param pool_seq_initializer: Optional[str], default None. When None, activation judge first,
 |                              xavier_normal end. When pool_seq_mode is in ('Weighted', 'Attention'), it works.
 |      
 |      Task Model Parameters:
 |          :param linear_if_bias: bool, default True.
 |          :param linear_l1: float, default 0.0.
 |          :param linear_l2: float, default 0.0.
 |          :param linear_initializer: Optional[str], default None. When None, activation judge first, xavier_normal end.
"""
```

## Code Examples

| Code Examples                                                          |
| ---------------------------------------------------------------------- |
| [TensorFlow](https://github.com/UlionTse/mlgb/tree/main/mlgb/examples) |
| [PyTorch](https://github.com/UlionTse/mlgb/tree/main/mlgb/examples)    |

## Citation

If you use this for research, please cite it using the following BibTeX entry. Thanks.

```bibtex
@misc{uliontse2020mlgb,
  author = {UlionTse},
  title = {MLGB is a library that includes many models of CTR Prediction & Recommender System by TensorFlow & PyTorch},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/UlionTse/mlgb}},
}
```
