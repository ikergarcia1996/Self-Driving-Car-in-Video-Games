import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


def get_resnet(model: int, pretrained: bool):
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

    Parameters:
    - embedded_size: Size of the feature vectors
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_fc: dropout probability for the output layer
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights
    """

    def __init__(
        self,
        embedded_size: int,
        dropout_cnn: float,
        dropout_fc: float,
        resnet: int,
        pretrained_resnet: bool,
    ):
        super(EncoderCNN, self).__init__()
        resnet = get_resnet(resnet, pretrained_resnet)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        modules_dropout = []
        for layer in modules:
            modules_dropout.append(layer)
            modules_dropout.append(nn.Dropout(dropout_cnn))

        self.resnet = nn.Sequential(*modules_dropout)
        self.fc = nn.Linear(resnet.fc.in_features, embedded_size)
        self.dropout = nn.Dropout(p=dropout_fc)
        self.bn = nn.BatchNorm1d(embedded_size, momentum=0.01)

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

    Parameters:
    - sequence_size: Length of each series of features
    """

    def __init__(self, sequence_size: int):
        super(PackFeatureVectors, self).__init__()
        self.sequence_size = sequence_size

    def forward(self, images):
        return images.view(
            int(images.size(0) / self.sequence_size), self.sequence_size, images.size(1)
        )


class EncoderRNN(nn.Module):
    """
    Extract feature vectors from input images (CNN)

    Input:
    torch.tensor [batch_size, sequence_size, embedded_size]

    Output:
    torch.tensor if bidirectional [batch_size, hidden_size*2]
                 else [batch_size, hidden_size]

     Parameters:
    - embedded_size: Size of the input feature vectors
    - hidden_size: LSTM hidden size
    - num_layers: number of layers in the LSTM
    - dropout_lstm: dropout probability for the LSTM
    - bidirectional: forward or bidirectional LSTM

    """

    def __init__(
        self,
        embedded_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional_lstm: bool,
        dropout_lstm: float,
    ):
        super(EncoderRNN, self).__init__()
        self.embed_size = embedded_size
        self.bidirectional_lstm = bidirectional_lstm
        self.lstm = nn.LSTM(
            embedded_size,
            hidden_size,
            num_layers,
            dropout=dropout_lstm,
            bidirectional=bidirectional_lstm,
            batch_first=True,
        )

    def forward(self, features):
        output, (h_n, c_n) = self.lstm(features)
        if self.bidirectional_lstm:
            x = torch.cat((h_n[-2], h_n[-1]), 1)
        else:
            x = self.fc(h_n[-1])
        return x

    def predict(self, features):
        with torch.no_grad():
            output, (h_n, c_n) = self.lstm(features)
            if self.bidirectional_lstm:
                x = torch.cat((h_n[-2], h_n[-1]), 1)
            else:
                x = self.fc(h_n[-1])
            return x


class OutputLayer(nn.Module):
    """
    Output linear layer that produces the predictions

    Input:
    torch.tensor [batch_size, hidden_size]

    Output:
    Forward: torch.tensor [batch_size, 12] (output values without softmax)
    Predict: torch.tensor [batch_size, 1] (index of the max value after softmax)

    Parameters:
    - hidden_size: Size of the input feature vectors
    - layers: list of integer, for each integer i a linear layer with i neurons will be added.
    """

    def __init__(self, hidden_size: int, layers: List[int] = None):
        super(OutputLayer, self).__init__()
        linear_layers = []
        if layers:
            linear_layers.append(hidden_size, layers[0])
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

    Parameters:
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights
    - sequence_size: Length of each series of features
    - embedded_size: Size of the feature vectors
    - hidden_size: LSTM hidden size
    - num_layers_lstm: number of layers in the LSTM
    - bidirectional_lstm: forward or bidirectional LSTM
    - layers_out: list of integer, for each integer i a linear layer with i neurons will be added.
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_fc: dropout probability for the output layer
    - dropout_lstm: dropout probability for the LSTM


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
        dropout_fc: float,
        dropout_lstm: float,
    ):
        super(TEDD1104, self).__init__()
