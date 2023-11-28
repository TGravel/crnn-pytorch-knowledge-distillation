# CRNN Pytorch with knowledge distillation

![python3.6](https://img.shields.io/badge/python-3.6-blue.svg)

## Disclaimer

This is an attempt/experiment to use vanilla soft and hard knowledge distillation on the CRNN architecture. 

If you want to just use the CRNN architecture I strongly recommend using the original repo [crnn-pytorch](https://github.com/GitYCC/crnn-pytorch) and article [https://ycc.idv.tw/crnn-ctc.html](https://ycc.idv.tw/crnn-ctc.html) by GitYCC on which this implementation is built on!


## CRNN + CTC

*From original repo:*

This is a Pytorch implementation of a Deep Neural Network for scene text recognition. It is based on the paper ["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (2016), Baoguang Shi et al."](http://arxiv.org/abs/1507.05717).

Blog article with more info: [https://ycc.idv.tw/crnn-ctc.html](https://ycc.idv.tw/crnn-ctc.html)

![crnn_structure](misc/crnn_structure.png)

### + KD

*Added by this fork:*

Knowledge Distillation according to the paper:

- ["Distilling the Knowledge in a Neural Network (2015), Hinton et al."](https://arxiv.org/abs/1503.02531)

Inspired by other papers (only single distillation):

- ["Well-Read Students
Learn Better: On the Importance of Pre-training Compact Models (2019), Turc et al."](https://arxiv.org/abs/1908.08962)

- ["Distilling Knowledge from Ensembles
of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition (2021), Gao et al."](https://arxiv.org/abs/2005.09310)

- ["Does Knowledge Distillation Really Work? (2021), Stanton et al."](https://arxiv.org/abs/2106.05945)


## Download datasets

[Synth90k / MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/)
```
@InProceedings{Jaderberg14c,
  author       = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman",
  title        = "Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition",
  booktitle    = "Workshop on Deep Learning, NIPS",
  year         = "2014",
}

@Article{Jaderberg16,
  author       = "Max Jaderberg and Karen Simonyan and Andrea Vedaldi and Andrew Zisserman",
  title        = "Reading Text in the Wild with Convolutional Neural Networks",
  journal      = "International Journal of Computer Vision",
  number       = "1",
  volume       = "116",
  pages        = "1--20",
  month        = "jan",
  year         = "2016",
}
```

### Process the following datasets with the corresponding notebook
[CocoTextv2](https://bgshih.github.io/cocotext/)
```
SE(3) Computer Vision Group at Cornell Tech licensed under a Creative Commons Attribution 4.0 License
```

[ICDAR2013](https://rrc.cvc.uab.es/?ch=2&com=downloads)
```
@INPROCEEDINGS{6628859,
  author={Karatzas, Dimosthenis and Shafait, Faisal and Uchida, Seiichi and Iwamura, Masakazu and Bigorda, Lluis Gomez i and Mestre, Sergi Robles and Mas, Joan and Mota, David Fernandez and Almazàn, Jon Almazàn and de las Heras, Lluís Pere},
  booktitle={2013 12th International Conference on Document Analysis and Recognition}, 
  title={ICDAR 2013 Robust Reading Competition}, 
  year={2013},
  volume={},
  number={},
  pages={1484-1493},
  doi={10.1109/ICDAR.2013.221}}
```

[IIIT5K](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)
```
@InProceedings{MishraBMVC12,
  author    = "Mishra, A. and Alahari, K. and Jawahar, C.~V.",
  title     = "Scene Text Recognition using Higher Order Language Priors",
  booktitle = "BMVC",
  year      = "2012",
}
```

## Pretrained Models

The weights for the best teacher and student I found over 30 epochs are provided.
Due to limited resources I could not investigate longer training times or more hyperparameters.




### Evaluate the model on the Synth90k dataset

```command
$ python src/evaluate.py
```

## Train/Distill your model

You can adjust hyper-parameters and set teachers as well as students in `./src/config.py`.

Train crnn models (like in the original repo):

```command
$ python src/train.py
```

Perform knowledge distillation via:

```command
$ python src/distill.py
```

Investigate multiple parameters for knowledge distillation via:

```command
$ python src/gridsearch.py
```


## Acknowledgement

This is a fork of the original repo, all credit goes to: [crnn-pytorch](https://github.com/GitYCC/crnn-pytorch)
