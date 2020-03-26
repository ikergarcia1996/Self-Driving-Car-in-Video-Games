import os
import json
from typing import List, Union

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet


def get_resnet(model: int, pretrained: bool) -> torchvision.models.resnet.ResNet:
    """
    Get resnet model

    Output:
     torchvision.models.resnet[18,34,50,101,152]

    Hyperparameters:
    - model: Resnet model from torchvision.models (number of layers): [18,34,50,101,152]
    - pretrained: Load model pretrained weights
    """
    if model == 18:
        return models.resnet18(pretrained=pretrained)
    elif model == 34:
        return models.resnet34(pretrained=pretrained)
    elif model == 50:
        return models.resnet50(pretrained=pretrained)
    elif model == 101:
        return models.resnet101(pretrained=pretrained)
    elif model == 152:
        return models.resnet152(pretrained=pretrained)

    raise ValueError(f"Resnet_{model} not found in torchvision.models")


class EncoderCNN(nn.Module):
    """
    Extract feature vectors from input images (CNN)

    Input:
     torch.tensor [batch_size, num_channels, H, W]

    Output:
     torch.tensor [batch_size, embedded_size]

    Hyperparameters:
    - embedded_size: Size of the feature vectors
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_cnn_out: dropout probability for the CNN representations (output layer)
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights
    """

    def __init__(
        self,
        embedded_size: int,
        dropout_cnn: float,
        dropout_cnn_out: float,
        resnet: int,
        pretrained_resnet: bool,
    ):
        super(EncoderCNN, self).__init__()
        resnet: models.resnet.ResNet = get_resnet(resnet, pretrained_resnet)
        modules: List[nn.Module] = list(resnet.children())[
            :-1
        ]  # delete the last fc layer.
        modules_dropout: List[Union[nn.Module, nn.Dropout]] = []
        for layer in modules:
            modules_dropout.append(layer)
            modules_dropout.append(nn.Dropout(dropout_cnn))

        self.resnet: nn.Module = nn.Sequential(*modules_dropout)
        self.fc: nn.Linear = nn.Linear(resnet.fc.in_features, embedded_size)
        self.dropout: nn.Dropout = nn.Dropout(p=dropout_cnn_out)
        self.bn: nn.BatchNorm1d = nn.BatchNorm1d(embedded_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.fc(features))
        features = self.dropout(features)
        return features

    def predict(self, images):
        with torch.no_grad():
            features = self.resnet(images)
            features = features.reshape(features.size(0), -1)
            features = self.bn(self.fc(features))
            return features


class PackFeatureVectors(nn.Module):
    """
    Reshape a list of features into a time distributed list of features. CNN ->  PackFeatureVectors -> RNN

    Input:
     torch.tensor [batch_size, embedded_size]

    Output:
     torch.tensor [batch_size/sequence_size, sequence_size, embedded_size]

    Hyperparameters:
    - sequence_size: Length of each series of features
    """

    def __init__(self, sequence_size: int):
        super(PackFeatureVectors, self).__init__()
        self.sequence_size: int = sequence_size

    def forward(self, images):
        return images.view(
            int(images.size(0) / self.sequence_size), self.sequence_size, images.size(1)
        )

    def predict(self, images):
        with torch.no_grad():
            return images.view(
                int(images.size(0) / self.sequence_size),
                self.sequence_size,
                images.size(1),
            )


class EncoderRNN(nn.Module):
    """
    Extract feature vectors from input images (CNN)

    Input:
     torch.tensor [batch_size, sequence_size, embedded_size]

    Output:
     torch.tensor if bidirectional [batch_size, hidden_size*2]
                 else [batch_size, hidden_size]

     Hyperparameters:
    - embedded_size: Size of the input feature vectors
    - hidden_size: LSTM hidden size
    - num_layers: number of layers in the LSTM
    - dropout_lstm: dropout probability for the LSTM
    - dropout_lstm_out: dropout probability for the LSTM representations (output layer)
    - bidirectional: forward or bidirectional LSTM

    """

    def __init__(
        self,
        embedded_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional_lstm: bool,
        dropout_lstm: float,
        dropout_lstm_out: float,
    ):
        super(EncoderRNN, self).__init__()

        self.lstm: nn.LSTM = nn.LSTM(
            embedded_size,
            hidden_size,
            num_layers,
            dropout=dropout_lstm,
            bidirectional=bidirectional_lstm,
            batch_first=True,
        )

        self.bidirectional_lstm = bidirectional_lstm

        self.dropout: nn.Dropout = nn.Dropout(p=dropout_lstm_out)

    def forward(self, features: torch.tensor):
        output, (h_n, c_n) = self.lstm(features)
        if self.bidirectional_lstm:
            x = torch.cat((h_n[-2], h_n[-1]), 1)
        else:
            x = h_n[-1]
        return self.dropout(x)

    def predict(self, features):
        with torch.no_grad():
            output, (h_n, c_n) = self.lstm(features)
            if self.bidirectional_lstm:
                x = torch.cat((h_n[-2], h_n[-1]), 1)
            else:
                x = h_n[-1]
            return x


class OutputLayer(nn.Module):
    """
    Output linear layer that produces the predictions

    Input:
     torch.tensor [batch_size, hidden_size]

    Output:
     Forward: torch.tensor [batch_size, 12] (output values without softmax)
     Predict: torch.tensor [batch_size, 1] (index of the max value after softmax)

    Hyperparameters:
    - hidden_size: Size of the input feature vectors
    - layers: list of integer, for each integer i a linear layer with i neurons will be added.
    """

    def __init__(self, hidden_size: int, layers: List[int] = None):
        super(OutputLayer, self).__init__()

        linear_layers: List[Union[nn.Linear, nn.ReLU]] = []
        if layers:
            linear_layers.append(nn.Linear(hidden_size, layers[0]))
            linear_layers.append(nn.ReLU())
            for i in range(1, len(layers)):
                linear_layers.append(nn.Linear(layers[i - 1], layers[i]))
                linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Linear(layers[-1], 9))

        else:
            linear_layers.append(nn.Linear(hidden_size, 9))

        self.linear = nn.Sequential(*linear_layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        return self.linear(inputs)

    def predict(self, inputs):
        with torch.no_grad():
            value, index = torch.max(self.softmax(self.linear(inputs)), 1)
            return index


class TEDD1104(nn.Module):
    """
    T.E.D.D. 1104 (https://nazizombiesplus.fandom.com/wiki/T.E.D.D.) is the neural network that learns
    how to drive in videogames. It has been develop with Grand Theft Auto V (GTAV) in mind. However
    it can learn how to drive in any videogame and if the model and controls are modified accordingly
    it can play any game. The model receive as input 5 consecutive images that have been captured
    with a fixed time interval between then (by default 1/10 seconds) and learn the correct
    key to push in the keyboard (None,W,A,S,D,WA,WD,SA,SD).

    T.E.D.D 1104 consists of 3 modules:
        [*] A CNN (Resnet) that extract features from the images
        [*] A RNN (LSTM) that generates a representation of the sequence of features from the CNN
        [*] A linear output layer that predicts the key to push.

    Input:
     torch.tensor [batch_size, num_channels, H, W]
     For efficiency the input input is not packed as sequence of 5 images, all the images in the batch will be
     encoded in the CNN and the features vectors will be packed as sequences of 5 vectors before feeding them to the
     RNN.

    Output:
     Forward: torch.tensor [batch_size, 12] (output values without softmax)
     Predict: torch.tensor [batch_size, 1] (index of the max value after softmax)

    Hyperparameters:
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights
    - sequence_size: Length of each series of features
    - embedded_size: Size of the feature vectors
    - hidden_size: LSTM hidden size
    - num_layers_lstm: number of layers in the LSTM
    - bidirectional_lstm: forward or bidirectional LSTM
    - layers_out: list of integer, for each integer i a linear layer with i neurons will be added.
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_cnn_out: dropout probability for the cnn features (output layer)
    - dropout_lstm: dropout probability for the LSTM
    - dropout_lstm_out: dropout probability for the LSTM features (output layer)

    """

    def __init__(
        self,
        resnet: int,
        pretrained_resnet: bool,
        sequence_size: int,
        embedded_size: int,
        hidden_size: int,
        num_layers_lstm: int,
        bidirectional_lstm: bool,
        layers_out: List[int],
        dropout_cnn: float,
        dropout_cnn_out: float,
        dropout_lstm: float,
        dropout_lstm_out: float,
    ):
        super(TEDD1104, self).__init__()

        # Remember hyperparameters.
        self.resnet: int = resnet
        self.pretrained_resnet: bool = pretrained_resnet
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.hidden_size: int = hidden_size
        self.num_layers_lstm: int = num_layers_lstm
        self.bidirectional_lstm: bool = bidirectional_lstm
        self.layers_out: List[int] = layers_out
        self.dropout_cnn: float = dropout_cnn
        self.dropout_cnn_out: float = dropout_cnn_out
        self.dropout_lstm: float = dropout_lstm
        self.dropout_lstm_out: float = dropout_lstm_out

        self.EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,
            dropout_cnn=dropout_cnn,
            dropout_cnn_out=dropout_cnn_out,
            resnet=resnet,
            pretrained_resnet=pretrained_resnet,
        )

        self.PackFeatureVectors = PackFeatureVectors(sequence_size=sequence_size)

        self.EncoderRNN = EncoderRNN(
            embedded_size=embedded_size,
            hidden_size=hidden_size,
            num_layers=num_layers_lstm,
            bidirectional_lstm=bidirectional_lstm,
            dropout_lstm=dropout_lstm,
            dropout_lstm_out=dropout_lstm_out,
        )

        self.OutputLayer = OutputLayer(
            hidden_size=int(hidden_size * 2) if bidirectional_lstm else hidden_size,
            layers=layers_out,
        )

    def forward(self, x):
        x = self.EncoderCNN(x)
        x = self.PackFeatureVectors(x)
        x = self.EncoderRNN(x)
        return self.OutputLayer(x)

    def predict(self, x):
        with torch.no_grad():
            x = self.EncoderCNN.predict(x)
            x = self.PackFeatureVectors.predict(x)
            x = self.EncoderRNN.predict(x)
            return self.OutputLayer.predict(x)


def save_model(model: TEDD1104, save_dir: str, fp16, amp_opt_level: str = None) -> None:
    """
    Save model to a directory. This function stores two files, the hyperparameters and the weights.

    Input:
     - model: TEDD1104 model to save
     - save_dir: directory where the model will be saved, if it doesn't exists we create it
     - amp: If the model uses FP16, Nvidia Apex AMP
     - amp_opt_level: If the model uses FP16, the AMP opt_level

    Output:

    """

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dict_hyperparams: dict = {
        "resnet": model.resnet,
        "pretrained_resnet": model.pretrained_resnet,
        "sequence_size": model.sequence_size,
        "embedded_size": model.embedded_size,
        "hidden_size": model.hidden_size,
        "num_layers_lstm": model.num_layers_lstm,
        "bidirectional_lstm": model.bidirectional_lstm,
        "layers_out": model.layers_out,
        "dropout_cnn": model.dropout_cnn,
        "dropout_cnn_out": model.dropout_cnn_out,
        "dropout_lstm": model.dropout_lstm,
        "dropout_lstm_out": model.dropout_lstm_out,
        "fp16": fp16,
        "amp_opt_level": amp_opt_level,
    }

    model_weights: dict = {
        "model": model.state_dict(),
        "amp": None if not fp16 else amp.state_dict(),
    }

    with open(os.path.join(save_dir, "model_hyperparameters.json"), "w+") as file:
        json.dump(dict_hyperparams, file)

    torch.save(obj=model_weights, f=os.path.join(save_dir, "model.bin"))


def load_model(save_dir: str, decive: torch.device) -> (TEDD1104, int):
    """
    Load a model from directory. The directory should contain a json with the model hyperparameters and a bin file
    with the model weights.

    Input:
     - save_dir: Directory where the model is stored

    Output:
    - TEDD1104 model
    - fp16: True if the model uses FP16 else False

    """

    with open(os.path.join(save_dir, "model_hyperparameters.json"), "r") as file:
        dict_hyperparams = json.load(file)

    model: TEDD1104 = TEDD1104(
        resnet=dict_hyperparams["resnet"],
        pretrained_resnet=dict_hyperparams["pretrained_resnet"],
        sequence_size=dict_hyperparams["sequence_size"],
        embedded_size=dict_hyperparams["embedded_size"],
        hidden_size=dict_hyperparams["hidden_size"],
        num_layers_lstm=dict_hyperparams["num_layers_lstm"],
        bidirectional_lstm=dict_hyperparams["bidirectional_lstm"],
        layers_out=dict_hyperparams["layers_out"],
        dropout_cnn=dict_hyperparams["dropout_cnn"],
        dropout_cnn_out=dict_hyperparams["dropout_cnn_out"],
        dropout_lstm=dict_hyperparams["dropout_lstm"],
        dropout_lstm_out=dict_hyperparams["dropout_lstm_out"],
    ).to(device=decive)

    model_weights = torch.load(f=os.path.join(save_dir, "model.bin"))

    if dict_hyperparams["fp16"]:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "The model you are trying to load uses FP16 training."
                "Please install apex from https://www.github.com/nvidia/apex"
            )

        model = amp.initialize(model, opt_level=dict_hyperparams["amp_opt_level"])
        amp.load_state_dict(model_weights["amp"])

    model.load_state_dict(model_weights["model"])

    return model, dict_hyperparams["fp16"]


def save_checkpoint(
    path: str,
    model: TEDD1104,
    optimizer_name: str,
    optimizer: torch.optim,
    acc_dev,
    epoch,
    fp16,
    opt_level=None,
) -> None:

    """
    Save a checkpoint that allows to continue training the model in the future

    Input:
     - path: path where the model is going to be saved
     - model: TEDD1104 model to save
     - optimizer_name: Name of the optimizer used for training: SGD or Adam
     - optimizer: Optimizer used for training
     - acc_dev: Accuracy of the model in the development set
     - epoch: Num of epoch used to train the model
     - amp: If the model uses FP16, Nvidia Apex AMP
     - amp_opt_level: If the model uses FP16, the AMP opt_level

    Output:
    """

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    dict_hyperparams: dict = {
        "sequence_size": model.sequence_size,
        "resnet": model.resnet,
        "pretrained_resnet": model.pretrained_resnet,
        "embedded_size": model.embedded_size,
        "hidden_size": model.hidden_size,
        "num_layers_lstm": model.num_layers_lstm,
        "bidirectional_lstm": model.bidirectional_lstm,
        "layers_out": model.layers_out,
        "dropout_cnn": model.dropout_cnn,
        "dropout_cnn_out": model.dropout_cnn_out,
        "dropout_lstm": model.dropout_lstm,
        "dropout_lstm_out": model.dropout_lstm_out,
        "fp16": fp16,
        "amp_opt_level": opt_level,
    }

    checkpoint = {
        "hyper_params": dict_hyperparams,
        "model": model.state_dict(),
        "optimizer_name": optimizer_name,
        "optimizer": optimizer.state_dict(),
        "acc_dev": acc_dev,
        "epoch": epoch,
        "amp": None if not fp16 else amp.state_dict(),
        "opt_level": opt_level,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str, device: torch.device
) -> (TEDD1104, str, torch.optim, float, int, bool, str):

    """
    Restore checkpoint

    Input:
    -path: path of the checkpoint to restore

    Output:
     - model: restored TEDD1104 model
     - optimizer_name: Name of the optimizer used for training: SGD or Adam
     - optimizer: Optimizer used for training
     - acc_dev: Accuracy of the model in the development set
     - epoch: Num of epoch used to train the model
     - fp16: true if the model uses fp16 else false
     - amp_opt_level: If the model uses FP16, the AMP opt_level

    Output:
    """

    checkpoint = torch.load(path)
    dict_hyperparams = checkpoint["hyper_params"]
    model_weights = checkpoint["model"]
    optimizer_name = checkpoint["optimizer_name"]
    optimizer_state = checkpoint["optimizer"]
    acc_dev = checkpoint["acc_dev"]
    epoch = checkpoint["acc_dev"]
    amp_state = checkpoint["amp"]
    opt_level = checkpoint["opt_level"]
    fp16 = dict_hyperparams["fp16"]

    model: TEDD1104 = TEDD1104(
        resnet=dict_hyperparams["resnet"],
        pretrained_resnet=dict_hyperparams["pretrained_resnet"],
        sequence_size=dict_hyperparams["sequence_size"],
        embedded_size=dict_hyperparams["embedded_size"],
        hidden_size=dict_hyperparams["hidden_size"],
        num_layers_lstm=dict_hyperparams["num_layers_lstm"],
        bidirectional_lstm=dict_hyperparams["bidirectional_lstm"],
        layers_out=dict_hyperparams["layers_out"],
        dropout_cnn=dict_hyperparams["dropout_cnn"],
        dropout_cnn_out=dict_hyperparams["dropout_cnn_out"],
        dropout_lstm=dict_hyperparams["dropout_lstm"],
        dropout_lstm_out=dict_hyperparams["dropout_lstm_out"],
    ).to(device=device)

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        raise ValueError(
            f"The optimizer you are trying to load is unknown: "
            f"Optimizer name {optimizer_name}. Available optimizers: SGD, Adam"
        )

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "The model you are trying to load uses FP16 training."
                "Please install apex from https://www.github.com/nvidia/apex"
            )

        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        amp.load_state_dict(amp_state)

    model.load_state_dict(model_weights)
    optimizer.load_state_dict(optimizer_state)

    return (
        model,
        optimizer_name,
        optimizer,
        acc_dev,
        epoch,
        fp16,
        opt_level,
    )
