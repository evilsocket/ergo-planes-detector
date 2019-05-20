`planes-detector` is an [ergo](https://github.com/evilsocket/ergo) based project that relies on a convolutional neural network to detect airplanes from satellite imagery.

<p align="center">
    <img src="https://i.imgur.com/imshZn6.png" width="400">
    <img src="https://i.imgur.com/Fbzedgs.png" width="400">
</p>

#### Training

Download the [PlanesNet dataset](https://www.kaggle.com/rhammell/planesnet) (the zip file with pictures) and extract it somewhere, then:

    ergo train /path/to/planes-detector --dataset /path/to/planesnet/pictures

This will make ergo preprocess the pictures, import them as a csv dataset and start the training algorithm (see [how to train on GPU](https://github.com/evilsocket/ergo#enable-gpu-support)) that can be monitored with `tensorboard` (`logs` folder).

After training is completed, you can view the model structure and how the accuracy and loss metrics changed during training with:

    ergo view /path/to/planes-detector

| Training | ROC/AUC |
|----------|---------|
![](https://raw.githubusercontent.com/evilsocket/ergo-planes-detector/master/history.png) | ![](https://raw.githubusercontent.com/evilsocket/ergo-planes-detector/master/roc.png) |

| Training | Validation | Testing |
|----------|------------|---------|
![](https://raw.githubusercontent.com/evilsocket/ergo-planes-detector/master/training_cm.png) | ![](https://raw.githubusercontent.com/evilsocket/ergo-planes-detector/master/validation_cm.png) | ![](https://raw.githubusercontent.com/evilsocket/ergo-planes-detector/master/test_cm.png) |

#### Evaluation

Once the training is completed, you can clean the project from temporary datasets and start a REST API server to test the model:

    cd /path/to/planes-detector
    ergo clean .
    ergo serve .

You can test the predictions with `curl`:

    curl http://127.0.0.1:8080/?x=/path/to/filename.jpg

Alternatively you can use the file `model.h5` (created inside the project folder after training) by loadeding it as you would normally do with Keras API for evaluation.

#### License

`planes-detector` was made with ♥  by [Simone Margaritelli](https://www.evilsocket.net/) and it is released under the GPL 3 license.
