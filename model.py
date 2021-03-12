from typing import List, Union, Optional
import os
import json
import math
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet
from torch.cuda.amp import GradScaler


class WeightedMseLoss(nn.Module):
    """
    Weighted mse loss columwise
    """

    _loss_log: torch.tensor

    def __init__(
        self,
        weights: List[float] = None,
        reduction: str = "mean",
    ):
        """
        INIT
        Input:
        - weights:  torch.tensor [3] Weights for each variable
        - reduction:  reduction method: sum or mean

        """
        assert reduction in ["sum", "mean"], (
            f"Reduction method: {reduction} not implemented. "
            f"Available reduction methods: [sum,mean]"
        )

        super(WeightedMseLoss, self).__init__()

        if weights is None:
            weights = [1.0, 1.0, 1.0]

        self.reduction = reduction
        self.register_buffer("weights", torch.tensor(weights))

        self._loss_log = torch.tensor([0.0, 0.0, 0.0], requires_grad=False)

    def forward(
        self,
        predicted: torch.tensor,
        target: torch.tensor,
    ) -> torch.tensor:

        """
        Input:
        - predicted: torch.tensor [batch_size, 3] Output from the model
        - predicted: torch.tensor [batch_size, 3] Gold values


        Output:
        -weighted_mse_loss columwise: torch.tensor  [1] if reduction == "mean"
                                                    [3] if reduction == "sum"
        """

        """
        DEBUG LOSS FUNCTION
            print(
                f"weights: {weights}\n"
                f"predicted: {predicted}\n"
                f"target: {target}\n"
                f"Distances: {weights * (predicted - target) ** 2}\n"
                f"Mean: {torch.mean(weights * (predicted - target) ** 2)}\n\n"
            )
        """
        if self.reduction == "mean":
            loss_per_joystick: torch.tensor = torch.mean(
                (predicted - target) ** 2, dim=0
            )
            self._loss_log = loss_per_joystick.detach()
            return torch.mean(self.weights * loss_per_joystick)
        else:
            loss_per_joystick: torch.tensor = torch.sum(
                (predicted - target) ** 2, dim=0
            )
            self._loss_log += loss_per_joystick.detach().to(self._loss_log.device)
            return self.weights * loss_per_joystick

    @property
    def loss_log(self):
        return self._loss_log.cpu()


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
        original_modules: List[nn.Module] = list(resnet.children())[:-1]
        modules: List[nn.Module, nn.Dropout] = []  # delete the last fc layer.

        for layer_no, layer in enumerate(original_modules):
            modules.append(layer)
            if layer_no + 1 != len(original_modules):
                modules.append(nn.Dropout(dropout_cnn))

        self.resnet: nn.Module = nn.Sequential(*modules)

        # if resnet.fc.in_features != embedded_size:
        self.fc: nn.Linear = nn.Linear(resnet.fc.in_features, embedded_size)
        self.dropout: nn.Dropout = nn.Dropout(p=dropout_cnn_out)

        self.bn: nn.BatchNorm1d = nn.BatchNorm1d(embedded_size, momentum=0.01)

    def forward(self, images: torch.tensor) -> torch.tensor:
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.dropout(self.fc(features))
        features = self.bn(features)
        return features

    def predict(self, images: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            features = self.resnet(images)
            features = features.reshape(features.size(0), -1)
            features = self.dropout(self.fc(features))
            features = self.bn(features)
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

    def forward(self, images: torch.tensor) -> torch.tensor:
        return images.view(
            int(images.size(0) / self.sequence_size), self.sequence_size, images.size(1)
        )

    def predict(self, images: torch.tensor) -> torch.tensor:
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

    def forward(self, features: torch.tensor) -> torch.tensor:
        output, (h_n, c_n) = self.lstm(features)
        if self.bidirectional_lstm:
            x = torch.cat((h_n[-2], h_n[-1]), 1)
        else:
            x = h_n[-1]
        return self.dropout(x)

    def predict(self, features: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            output, (h_n, c_n) = self.lstm(features)
            if self.bidirectional_lstm:
                x = torch.cat((h_n[-2], h_n[-1]), 1)
            else:
                x = h_n[-1]
            return x


class EncoderTransformer(nn.Module):
    """
    Extract feature vectors from input images (Transformer Encoder)

    Input:
     torch.tensor [batch_size, sequence_size, embedded_size]

    Output:
     torch.tensor [batch_size, hidden_size]

     Hyperparameters:
    - embedded_size: Size of the input feature vectors
    - nhead: LSTM hidden size
    - num_layers_transformer: number of transformer layers in the encoder
    - dropout_transformer_out: dropout probability for the transformer features (output layer)

    """

    def __init__(
        self,
        embedded_size: int,
        nhead: int,
        num_layers: int,
        dropout_out: float,
    ):
        super(EncoderTransformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedded_size, nhead=nhead
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.dropout: nn.Dropout = nn.Dropout(p=dropout_out)

    def forward(self, features: torch.tensor):
        x = self.transformer_encoder(features)
        x = torch.flatten(x, start_dim=1)
        return self.dropout(x)

    def predict(self, features):
        with torch.no_grad():
            x = self.transformer_encoder(features)
            return torch.flatten(x, start_dim=1)


class PositionalEncoding(nn.Module):
    """
    Add positional encodings to the transformer input features

    Input:
     torch.tensor [batch_size, sequence_size, embedded_size]

    Output:
     torch.tensor [batch_size, sequence_size, embedded_size]

     Hyperparameters:
    - embedded_size: Size of the input feature vectors
    - dropout: dropout probability for the embeddings

    """

    def __init__(self, d_model, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(5, d_model)
        position = torch.arange(0, 5, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.positional_encoding[: x.size(0), :]
        return self.dropout(x)

    def predict(self, x: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            return x + self.positional_encoding[: x.size(0), :]


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

        linear_layers: List[Union[nn.Linear, nn.LeakyReLU]] = []
        if layers:
            linear_layers.append(nn.Linear(hidden_size, layers[0]))
            linear_layers.append(nn.LeakyReLU())
            for i in range(1, len(layers)):
                linear_layers.append(nn.Linear(layers[i - 1], layers[i]))
                linear_layers.append(nn.LeakyReLU())
            linear_layers.append(nn.Linear(layers[-1], 3))

        else:
            linear_layers.append(nn.Linear(hidden_size, 3))

        self.linear = nn.Sequential(*linear_layers)

        # self.sigmoid: nn.Sigmoid = (
        #    nn.Sigmoid()
        # )  # We want the output to be in range [-1,1], sigmoid will ensure it.

    def forward(self, inputs):
        # return -1.0 + 2.0 * self.sigmoid(self.linear(inputs))
        return self.linear(inputs)

    def predict(self, inputs):
        with torch.no_grad():
            # return -1.0 + 2.0 * self.sigmoid(self.linear(inputs))
            return self.linear(inputs)


class TEDD1104LSTM(nn.Module):
    """
    T.E.D.D. 1104 (https://nazizombiesplus.fandom.com/wiki/T.E.D.D.) is the neural network that learns
    how to drive in videogames. It has been develop with Grand Theft Auto V (GTAV) in mind. However
    it can learn how to drive in any videogame and if the model and controls are modified accordingly
    it can play any game. The model receive as input 5 consecutive images that have been captured
    with a fixed time interval between then (by default 1/10 seconds) and learns the correct
    controller input.

    T.E.D.D 1104 consists of 3 modules:
        [*] A CNN (Resnet) that extract features from the images
        [*] A RNN (LSTM) that generates a representation of the sequence of features from the CNN
        [*] A linear output layer that predicts the controller input.

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
        super(TEDD1104LSTM, self).__init__()

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

        self.EncoderCNN: EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,
            dropout_cnn=dropout_cnn,
            dropout_cnn_out=dropout_cnn_out,
            resnet=resnet,
            pretrained_resnet=pretrained_resnet,
        )

        self.PackFeatureVectors: PackFeatureVectors = PackFeatureVectors(
            sequence_size=sequence_size
        )

        self.EncoderRNN: EncoderRNN = EncoderRNN(
            embedded_size=embedded_size,
            hidden_size=hidden_size,
            num_layers=num_layers_lstm,
            bidirectional_lstm=bidirectional_lstm,
            dropout_lstm=dropout_lstm,
            dropout_lstm_out=dropout_lstm_out,
        )

        self.OutputLayer: OutputLayer = OutputLayer(
            hidden_size=int(hidden_size * 2) if bidirectional_lstm else hidden_size,
            layers=layers_out,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.EncoderCNN(x)
        x = self.PackFeatureVectors(x)
        x = self.EncoderRNN(x)
        return self.OutputLayer(x)

    def predict(self, x: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            x = self.EncoderCNN.predict(x)
            x = self.PackFeatureVectors.predict(x)
            x = self.EncoderRNN.predict(x)
            return self.OutputLayer.predict(x)

    def save_model(self, save_dir: str) -> None:
        """
        Save model to a directory. This function stores two files, the hyperparameters and the weights.

        Input:
         - model: TEDD1104 model to save
         - save_dir: directory where the model will be saved, if it doesn't exists we create it

        Output:

        """

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dict_hyperparams: dict = {
            "resnet": self.resnet,
            "pretrained_resnet": self.pretrained_resnet,
            "sequence_size": self.sequence_size,
            "embedded_size": self.embedded_size,
            "hidden_size": self.hidden_size,
            "num_layers_lstm": self.num_layers_lstm,
            "bidirectional_lstm": self.bidirectional_lstm,
            "layers_out": self.layers_out,
            "dropout_cnn": self.dropout_cnn,
            "dropout_cnn_out": self.dropout_cnn_out,
            "dropout_lstm": self.dropout_lstm,
            "dropout_lstm_out": self.dropout_lstm_out,
        }

        model_weights: dict = {
            "model": self.state_dict(),
        }

        with open(os.path.join(save_dir, "model_hyperparameters.json"), "w+") as file:
            json.dump(dict_hyperparams, file)

        torch.save(obj=model_weights, f=os.path.join(save_dir, "model.bin"))

    def save_checkpoint(
        self,
        path: str,
        optimizer_name: str,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        running_loss: float,
        loss_per_joystick: torch.tensor,
        total_batches: int,
        total_training_examples: int,
        min_loss_dev: float,
        epoch: int,
        scaler: Optional[GradScaler],
    ) -> None:

        """
        Save a checkpoint that allows to continue training the model in the future

        Input:
         - path: path where the model is going to be saved
         - optimizer_name: Name of the optimizer used for training: SGD or Adam
         - optimizer: Optimizer used for training
         - scheduler: Scheduler used for training
         - running_loss: Current running loss in the training set
         - loss_per_joystick: Current running loss per input joystick/trigger in the training set
         - total_batches: Total batches used for training the model
         - total_training_examples: Total training examples used for training the model
         - min_loss_dev: Min loss in the development set
         - epoch: Num of epoch used to train the model
         - scaler: The scaler used for FP16 training

        Output:
        """

        dict_hyperparams: dict = {
            "sequence_size": self.sequence_size,
            "resnet": self.resnet,
            "pretrained_resnet": self.pretrained_resnet,
            "embedded_size": self.embedded_size,
            "hidden_size": self.hidden_size,
            "num_layers_lstm": self.num_layers_lstm,
            "bidirectional_lstm": self.bidirectional_lstm,
            "layers_out": self.layers_out,
            "dropout_cnn": self.dropout_cnn,
            "dropout_cnn_out": self.dropout_cnn_out,
            "dropout_lstm": self.dropout_lstm,
            "dropout_lstm_out": self.dropout_lstm_out,
        }

        checkpoint = {
            "hyper_params": dict_hyperparams,
            "model": self.state_dict(),
            "optimizer_name": optimizer_name,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "running_loss": running_loss,
            "loss_per_joystick": loss_per_joystick,
            "total_batches": total_batches,
            "total_training_examples": total_training_examples,
            "min_loss_dev": min_loss_dev,
            "epoch": epoch,
            "scaler": None if not scaler else scaler.state_dict(),
        }

        torch.save(checkpoint, path)


class TEDD1104Transformer(nn.Module):
    """
    T.E.D.D. 1104 (https://nazizombiesplus.fandom.com/wiki/T.E.D.D.) is the neural network that learns
    how to drive in videogames. It has been develop with Grand Theft Auto V (GTAV) in mind. However
    it can learn how to drive in any videogame and if the model and controls are modified accordingly
    it can play any game. The model receive as input 5 consecutive images that have been captured
    with a fixed time interval between then (by default 1/10 seconds) and learns the correct
    controller input.

    T.E.D.D 1104 consists of 3 modules:
        [*] A CNN (Resnet) that extract features from the images
        [*] A Transformer that generates a representation of the sequence of features from the CNN
        [*] A linear output layer that predicts the controller input.

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
    - nhead: number of heads for the transformer layer
    - num_layers_transformer: number of transformer layers in the encoder
    - layers_out: list of integer, for each integer i a linear layer with i neurons will be added.
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_cnn_out: dropout probability for the cnn features (output layer)
    - positional_embeddings_dropout: dropout probability for the transformer input embeddings
    - dropout_transformer_out: dropout probability for the transformer features (output layer)

    """

    def __init__(
        self,
        resnet: int,
        pretrained_resnet: bool,
        sequence_size: int,
        embedded_size: int,
        nhead: int,
        num_layers_transformer: int,
        layers_out: List[int],
        dropout_cnn: float,
        dropout_cnn_out: float,
        positional_embeddings_dropout: float,
        dropout_transformer_out: float,
    ):
        super(TEDD1104Transformer, self).__init__()

        # Remember hyperparameters.
        self.resnet: int = resnet
        self.pretrained_resnet: bool = pretrained_resnet
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.nhead: int = nhead
        self.num_layers_transformer: int = num_layers_transformer
        self.layers_out: List[int] = layers_out
        self.dropout_cnn: float = dropout_cnn
        self.dropout_cnn_out: float = dropout_cnn_out
        self.positional_embeddings_dropout: float = positional_embeddings_dropout
        self.dropout_transformer_out: float = dropout_transformer_out

        self.EncoderCNN: EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,
            dropout_cnn=dropout_cnn,
            dropout_cnn_out=dropout_cnn_out,
            resnet=resnet,
            pretrained_resnet=pretrained_resnet,
        )

        self.PackFeatureVectors: PackFeatureVectors = PackFeatureVectors(
            sequence_size=sequence_size
        )

        self.PositionalEncoding = PositionalEncoding(
            d_model=embedded_size, dropout=self.positional_embeddings_dropout
        )

        self.EncoderTransformer: EncoderTransformer = EncoderTransformer(
            embedded_size=embedded_size,
            nhead=nhead,
            num_layers=num_layers_transformer,
            dropout_out=dropout_transformer_out,
        )

        self.OutputLayer: OutputLayer = OutputLayer(
            hidden_size=embedded_size * sequence_size,
            layers=layers_out,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.EncoderCNN(x)
        x = self.PackFeatureVectors(x)
        x = self.PositionalEncoding(x)
        x = self.EncoderTransformer(x)
        return self.OutputLayer(x)

    def predict(self, x: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            x = self.EncoderCNN.predict(x)
            x = self.PackFeatureVectors.predict(x)
            x = self.PositionalEncoding.predict(x)
            x = self.EncoderTransformer.predict(x)
            return self.OutputLayer.predict(x)

    def save_model(self, save_dir: str) -> None:
        """
        Save model to a directory. This function stores two files, the hyperparameters and the weights.

        Input:
         - model: TEDD1104 model to save
         - save_dir: directory where the model will be saved, if it doesn't exists we create it

        Output:

        """

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dict_hyperparams: dict = {
            "resnet": self.resnet,
            "pretrained_resnet": self.pretrained_resnet,
            "sequence_size": self.sequence_size,
            "embedded_size": self.embedded_size,
            "nhead": self.nhead,
            "num_layers_transformer": self.num_layers_transformer,
            "layers_out": self.layers_out,
            "dropout_cnn": self.dropout_cnn,
            "dropout_cnn_out": self.dropout_cnn_out,
            "positional_embeddings_dropout": self.positional_embeddings_dropout,
            "dropout_transformer_out": self.dropout_transformer_out,
        }

        model_weights: dict = {
            "model": self.state_dict(),
        }

        with open(os.path.join(save_dir, "model_hyperparameters.json"), "w+") as file:
            json.dump(dict_hyperparams, file)

        torch.save(obj=model_weights, f=os.path.join(save_dir, "model.bin"))

    def save_checkpoint(
        self,
        path: str,
        optimizer_name: str,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        running_loss: float,
        loss_per_joystick: torch.tensor,
        total_batches: int,
        total_training_examples: int,
        min_loss_dev: float,
        epoch: int,
        scaler: Optional[GradScaler],
    ) -> None:

        """
        Save a checkpoint that allows to continue training the model in the future

        Input:
         - path: path where the model is going to be saved
         - optimizer_name: Name of the optimizer used for training: SGD or Adam
         - optimizer: Optimizer used for training
         - scheduler: Scheduler used for training
         - running_loss: Current running loss in the training set
         - loss_per_joystick: Current running loss per input joystick/trigger in the training set
         - total_batches: Total batches used for training the model
         - total_training_examples: Total training examples used for training the model
         - min_loss_dev: Min loss in the development set
         - epoch: Num of epoch used to train the model
         - scaler: The scaler used for FP16 training


        Output:
        """

        dict_hyperparams: dict = {
            "resnet": self.resnet,
            "pretrained_resnet": self.pretrained_resnet,
            "sequence_size": self.sequence_size,
            "embedded_size": self.embedded_size,
            "nhead": self.nhead,
            "num_layers_transformer": self.num_layers_transformer,
            "layers_out": self.layers_out,
            "dropout_cnn": self.dropout_cnn,
            "dropout_cnn_out": self.dropout_cnn_out,
            "positional_embeddings_dropout": self.positional_embeddings_dropout,
            "dropout_transformer_out": self.dropout_transformer_out,
        }

        checkpoint = {
            "hyper_params": dict_hyperparams,
            "model": self.state_dict(),
            "optimizer_name": optimizer_name,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "running_loss": running_loss,
            "loss_per_joystick": loss_per_joystick,
            "total_batches": total_batches,
            "total_training_examples": total_training_examples,
            "min_loss_dev": min_loss_dev,
            "epoch": epoch,
            "scaler": None if not scaler else scaler.state_dict(),
        }

        torch.save(checkpoint, path)


TEDD1104 = Union[TEDD1104LSTM, TEDD1104Transformer]


def load_model(
    save_dir: str, device: torch.device
) -> Union[TEDD1104LSTM, TEDD1104Transformer]:
    """
    Load a model from directory. The directory should contain a json with the model hyperparameters and a bin file
    with the model weights.

    Input:
     - save_dir: Directory where the model is stored

    Output:
    - TEDD1104 model

    """

    with open(os.path.join(save_dir, "model_hyperparameters.json"), "r") as file:
        dict_hyperparams = json.load(file)

    if "num_layers_lstm" in dict_hyperparams:
        print(f"Loading TEDD1104LSTM model")
        model: TEDD1104LSTM = TEDD1104LSTM(
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

    else:
        print(f"Loading TEDD1104Transformer model")
        model: TEDD1104Transformer = TEDD1104Transformer(
            resnet=dict_hyperparams["resnet"],
            pretrained_resnet=dict_hyperparams["pretrained_resnet"],
            sequence_size=dict_hyperparams["sequence_size"],
            embedded_size=dict_hyperparams["embedded_size"],
            nhead=dict_hyperparams["nhead"],
            num_layers_transformer=dict_hyperparams["num_layers_transformer"],
            layers_out=dict_hyperparams["layers_out"],
            dropout_cnn=dict_hyperparams["dropout_cnn"],
            dropout_cnn_out=dict_hyperparams["dropout_cnn_out"],
            positional_embeddings_dropout=dict_hyperparams[
                "positional_embeddings_dropout"
            ],
            dropout_transformer_out=dict_hyperparams["dropout_transformer_out"],
        ).to(device=device)

    model_weights = torch.load(f=os.path.join(save_dir, "model.bin"))
    model.load_state_dict(model_weights["model"])

    return model


def load_checkpoint(
    path: str, device: torch.device
) -> (
    Union[TEDD1104LSTM, TEDD1104Transformer],
    str,
    torch.optim,
    torch.optim.lr_scheduler,
    float,
    torch.tensor,
    int,
    int,
    float,
    int,
    Union[GradScaler, None],
):

    """
    Restore checkpoint

    Input:
    -path: path of the checkpoint to restore

    Output:
     - model: restored TEDD1104 model
     - optimizer_name: Name of the optimizer used for training: SGD or Adam
     - scheduler: Scheduler used for training
     - running_loss: Running loss of the model
     - loss_per_joystick: Running loss per joystick/trigger of the model
     - total_batches: Total batches used for training
     - total_training_examples: Training examples used for training
     - min_loss_dev: Min loss in development set
     - epoch: Num of epoch used to train the model
     - scaler: If the model uses FP16, the scaler used for training
    """

    checkpoint = torch.load(path)
    dict_hyperparams = checkpoint["hyper_params"]
    model_weights = checkpoint["model"]
    optimizer_name = checkpoint["optimizer_name"]
    optimizer_state = checkpoint["optimizer"]
    min_loss_dev = checkpoint["min_loss_dev"]
    epoch = checkpoint["epoch"]
    scaler_state = checkpoint["scaler"]

    if "num_layers_lstm" in dict_hyperparams:
        print(f"Loading TEDD1104LSTM model")
        model: TEDD1104LSTM = TEDD1104LSTM(
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

    else:
        print(f"Loading TEDD1104Transformer model")
        model: TEDD1104Transformer = TEDD1104Transformer(
            resnet=dict_hyperparams["resnet"],
            pretrained_resnet=dict_hyperparams["pretrained_resnet"],
            sequence_size=dict_hyperparams["sequence_size"],
            embedded_size=dict_hyperparams["embedded_size"],
            nhead=dict_hyperparams["nhead"],
            num_layers_transformer=dict_hyperparams["num_layers_transformer"],
            layers_out=dict_hyperparams["layers_out"],
            dropout_cnn=dict_hyperparams["dropout_cnn"],
            dropout_cnn_out=dict_hyperparams["dropout_cnn_out"],
            positional_embeddings_dropout=dict_hyperparams[
                "positional_embeddings_dropout"
            ],
            dropout_transformer_out=dict_hyperparams["dropout_transformer_out"],
        ).to(device=device)

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-04)
    else:
        raise ValueError(
            f"The optimizer you are trying to load is unknown: "
            f"Optimizer name {optimizer_name}. Available optimizers: SGD, Adam"
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    model.load_state_dict(model_weights)
    optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint["scheduler"]
    scheduler.load_state_dict(scheduler_state)

    running_loss = checkpoint["running_loss"]

    total_training_examples = checkpoint["total_training_examples"]

    total_batches = checkpoint["total_batches"]

    scaler: Optional[GradScaler]
    if scaler_state:
        scaler = GradScaler()
        scaler.load_state_dict(scaler_state)
    else:
        scaler = None

    try:
        loss_per_joystick = checkpoint["loss_per_joystick"]
    except KeyError:
        print(
            f"{path} is a legacy checkpoint, we will initialize loss_per_joystick as tensor of zeros."
        )
        loss_per_joystick = torch.tensor([0.0, 0.0, 0.0], requires_grad=False)

    return (
        model,
        optimizer_name,
        optimizer,
        scheduler,
        running_loss,
        loss_per_joystick,
        total_batches,
        total_training_examples,
        min_loss_dev,
        epoch,
        scaler,
    )
