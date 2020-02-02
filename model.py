import torch
import torch.nn as nn
import torchvision.models as models


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
    Initialization parameters:
    - embedded_size: Size of the feature vectors
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_fc: dropout probability for the output layer
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights

    Input:
    torch.tensor [batch_size, num_channels, H, W]

    Output:
    torch.tensor [batch_size, embedded_size]

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
    Initialization parameters:
    - sequence_size: Length of each series of features

    Input:
     torch.tensor [batch_size, embedded_size]

    output:
    torch.tensor [batch_size/sequence_size, sequence_size, embedded_size]
    
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
    Initialization parameters:
    - embedded_size: Size of the input feature vectors
    - hidden_size: LSTM hidden size
    - num_layers: number of layers in the LSTM
    - dropout_lstm: dropout probability for the LSTM
    - bidirectional: forward or bidirectional LSTM
    Input:
    torch.tensor [batch_size, sequence_size, embedded_size]

    Output:
    torch.tensor if bidirectional [batch_size, hidden_size*2]
                 else [batch_size, hidden_size]

    """

    def __init__(
        self,
        embedded_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional_lstm,
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
