# cirfar10_ae_vgg
It's an experiment using an Autoencoder as feature extractor and feed the hidden features into vgg16 model.
In order to examine if it can improve the vgg16 result by using the hidden features.

To train end to end model

    $python train_end2end.py

To train separated model

    $python train_separated.py

To train VGG_only model

    $python train_vgg_only.py

To train VGG-like encoder with fully connected classifier from vgg16 model(end to end model)

    $python train_end2end_2.py
