"""
T.E.D.D. 1104 (Reference to the robot that drives a bus is Call of Duty: Black Ops II tranzit mode
https://nazizombiesplus.fandom.com/wiki/T.E.D.D.)
is the neural network that learns how to drive in video-games.
It has been developed with Grand Theft Auto V (GTAV) in mind.
However, it can learn how to drive in any video-game and if the game controls are modified accordingly.
T.E.D.D. 1104 is a supervised learning model, it learns from the examples recorded when human players play the game.

Additionally, 'TEDD1104 for image reordering' predicts the correct order of images in a shuffled sequence.
It is intended as a pretraining task before training in the self-driving objetive.


Developed by Iker García-Ferrero:
Website: https://ikergarcia1996.github.io/Iker-Garcia-Ferrero/
Github: https://github.com/ikergarcia1996

Some of the code for input handling and image recording was developed with the help of Aiden Yerga:
Github: https://github.com/aidenyg

Project GitHub repository:
https://github.com/ikergarcia1996/Self-Driving-Car-in-Video-Games
"""

from typing import List
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics


class WeightedMseLoss(nn.Module):
    """
    Weighted mse loss columnwise
    """

    def __init__(
        self,
        weights: List[float] = None,
        reduction: str = "mean",
    ):
        """
        INIT

        :param List[float] weights: List of weights for each joystick
        :param str reduction: "mean" or "sum"
        """

        assert reduction in ["sum", "mean"], (
            f"Reduction method: {reduction} not implemented. "
            f"Available reduction methods: [sum,mean]"
        )

        super(WeightedMseLoss, self).__init__()

        self.reduction = reduction
        if not weights:
            weights = [1.0, 1.0]
        weights = torch.tensor(weights)
        weights.requires_grad = False

        self.register_buffer("weights", weights)

    def forward(
        self,
        predicted: torch.tensor,
        target: torch.tensor,
    ) -> torch.tensor:

        """
        Forward pass

        :param torch.tensor predicted: Predicted values [batch_size, 2]
        :param torch.tensor target: Target values [batch_size, 2]
        :return: Loss [1] if reduction is "mean" else [2]
        """

        if self.reduction == "mean":
            loss_per_joystick: torch.tensor = torch.mean(
                (predicted - target) ** 2, dim=0
            )
            return torch.mean(self.weights * loss_per_joystick)
        else:
            loss_per_joystick: torch.tensor = torch.sum(
                (predicted - target) ** 2, dim=0
            )
            return self.weights * loss_per_joystick


class CrossEntropyLoss(torch.nn.Module):
    """
    Weighted CrossEntropyLoss
    """

    def __init__(
        self,
        weights: List[float] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        INIT

        :param List[float] weights: List of weights for each key combination [9]
        :param str reduction: "mean" or "sum"
        :param float label_smoothing: A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss
        """

        assert reduction in ["sum", "mean"], (
            f"Reduction method: {reduction} not implemented. "
            f"Available reduction methods: [sum,mean]"
        )

        super(CrossEntropyLoss, self).__init__()

        self.reduction = reduction
        if weights:
            weights = torch.tensor(weights)
            weights.requires_grad = False

        self.register_buffer("weights", weights)

        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss(
            reduction=reduction,
            weight=weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        predicted: torch.tensor,
        target: torch.tensor,
    ) -> torch.tensor:

        """
        Forward pass

        :param torch.tensor predicted: Predicted values [batch_size, 9]
        :param torch.tensor target: Target values [batch_size]
        :return: Loss [1] if reduction is "mean" else [9]
        """
        return self.CrossEntropyLoss(predicted, target)


class CrossEntropyLossImageReorder(torch.nn.Module):
    """
    Weighted CrossEntropyLoss for Image Reordering
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
    ):
        """
        INIT

        :param float label_smoothing: A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss
        """

        super(CrossEntropyLossImageReorder, self).__init__()

        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        predicted: torch.tensor,
        target: torch.tensor,
    ) -> torch.tensor:

        """
        Forward pass

        :param torch.tensor predicted: Predicted values [batch_size, 5]
        :param torch.tensor target: Target values [batch_size]
        :return: Loss [1]
        """

        return self.CrossEntropyLoss(predicted.view(-1, 5), target.view(-1).long())


class ImageReorderingAccuracy(torchmetrics.Metric):
    """
    Image Reordering Accuracy Metric
    """

    def __init__(self, dist_sync_on_step=False):
        """
        INIT

        :param bool dist_sync_on_step: If True, the metric will be synchronized on step
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric with the given predictions and targets

        :param torch.Tensor preds: Predictions [batch_size, 5]
        :param torch.Tensor target: Target values [batch_size]
        """
        assert (
            preds.size() == target.size()
        ), f"Pred sise: {preds.size()} != Target size: {target.size()}"

        self.correct += torch.sum(torch.all(preds == target, dim=-1))
        self.total += target.size(0)

    def compute(self):
        return self.correct.float() / self.total


def get_cnn(cnn_model_name: str, pretrained: bool) -> (torchvision.models, int):
    """
    Get a CNN model from torchvision.models (https://pytorch.org/vision/stable/models.html)

    :param str cnn_model_name: Name of the CNN model from torchvision.models
    :param bool pretrained: If True, the model will be loaded with pretrained weights
    :return: CNN model, last layer output size
    """

    # Get the CNN model
    cnn_call_method = getattr(models, cnn_model_name)
    cnn_model = cnn_call_method(pretrained=pretrained)

    # Remove classification layer
    _ = cnn_model._modules.popitem()
    cnn_model = nn.Sequential(*list(cnn_model.children()))

    # Test output_size of last layer of the CNN (Not the most efficient way, but it works)
    features = cnn_model(torch.zeros((1, 3, 270, 480), dtype=torch.float32))
    output_size: int = features.reshape(features.size(0), -1).size(1)

    return cnn_model, output_size


class EncoderCNN(nn.Module):
    """
    Encoder CNN, extracts features from the input images

    For efficiency the input is a single sequence of [sequence_size*batch_size] images,
    the output of the CNN will be packed as sequences of sequence_size vectors.
    """

    def __init__(
        self,
        embedded_size: int,
        dropout_cnn_out: float,
        cnn_model_name: str,
        pretrained_cnn: bool,
        sequence_size: int = 5,
    ):
        """
        INIT

        :param int embedded_size: Size of the output embedding
        :param float dropout_cnn_out: Dropout rate for the output of the CNN
        :param str cnn_model_name: Name of the CNN model from torchvision.models
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights
        :param int sequence_size: Size of the sequence of images
        """
        super(EncoderCNN, self).__init__()

        self.embedded_size = embedded_size
        self.cnn_model_name = cnn_model_name
        self.dropout_cnn_out = dropout_cnn_out
        self.pretrained_cnn = pretrained_cnn

        self.cnn, self.cnn_output_size = get_cnn(
            cnn_model_name=cnn_model_name, pretrained=pretrained_cnn
        )

        self.dp = nn.Dropout(p=dropout_cnn_out)
        self.dense = nn.Linear(self.cnn_output_size, self.cnn_output_size)
        self.layer_norm = nn.LayerNorm(self.cnn_output_size, eps=1e-05)

        self.decoder = nn.Linear(self.cnn_output_size, self.embedded_size)
        self.bias = nn.Parameter(torch.zeros(self.embedded_size))
        self.decoder.bias = self.bias
        self.gelu = nn.GELU()
        self.sequence_size = sequence_size

    def forward(self, images: torch.tensor) -> torch.tensor:
        """
        Forward pass
        :param torch.tensor images: Input images [batch_size * sequence_size, 3, 270, 480]
        :return: Output embedding [batch_size, sequence_size, embedded_size]
        """
        features = self.cnn(images)
        features = features.reshape(features.size(0), -1)

        """
        Reshapes the features from the CNN into a time distributed format
        """

        features = features.view(
            int(features.size(0) / self.sequence_size),
            self.sequence_size,
            features.size(1),
        )

        features = self.dp(features)
        features = self.dense(features)
        features = self.gelu(features)
        features = self.layer_norm(features)
        features = self.decoder(features)
        return features

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class EncoderRNN(nn.Module):
    """
    Extracts features from the input sequence using an RNN
    """

    def __init__(
        self,
        embedded_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional_lstm: bool,
        dropout_lstm: float,
    ):

        """
        INIT
        :param int embedded_size: Size of the input feature vectors
        :param int hidden_size: LSTM hidden size
        :param int num_layers: number of layers in the LSTM
        :param bool bidirectional_lstm: forward or bidirectional LSTM
        :param float dropout_lstm: dropout probability for the LSTM
        """
        super(EncoderRNN, self).__init__()

        self.embedded_size = embedded_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional_lstm = bidirectional_lstm
        self.dropout_lstm = dropout_lstm

        self.lstm: nn.LSTM = nn.LSTM(
            embedded_size,
            hidden_size,
            num_layers,
            dropout=dropout_lstm,
            bidirectional=bidirectional_lstm,
            batch_first=True,
        )

        self.bidirectional_lstm = bidirectional_lstm

    def forward(self, features: torch.tensor) -> torch.tensor:
        """
        Forward pass
        :param torch.tensor features: Input features [batch_size, sequence_size, embedded_size]
        :return: Output features [batch_size, hidden_size*2 if bidirectional else hidden_size]
        """
        output, (h_n, c_n) = self.lstm(features)
        if self.bidirectional_lstm:
            x = torch.cat((h_n[-2], h_n[-1]), 1)
        else:
            x = h_n[-1]
        return x


class PositionalEmbedding(nn.Module):
    """
    Add positional encodings to the transformer input features
    """

    def __init__(
        self,
        sequence_length: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        """
        INIT
        :param int sequence_length: Length of the input sequence
        :param int d_model: Size of the input feature vectors
        :param float dropout: dropout probability for the embeddings
        """
        super(PositionalEmbedding, self).__init__()

        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = dropout

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(self.sequence_length, d_model).float()
        pe.requires_grad = True
        pe = pe.unsqueeze(0)
        self.pe = torch.nn.Parameter(pe)
        torch.nn.init.normal_(self.pe, std=0.02)
        self.LayerNorm = nn.LayerNorm(self.d_model, eps=1e-05)
        self.dp = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        :param torch.tensor x: Input features [batch_size, sequence_size, embedded_size]
        :return: Output features [batch_size, sequence_size, embedded_size]
        """
        pe = self.pe[:, : x.size(1)]
        x = pe + x
        x = self.LayerNorm(x)
        x = self.dp(x)
        return x


class EncoderTransformer(nn.Module):
    """
    Extracts features from the input sequence using a Transformer
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        sequence_length: int = 5,
    ):
        """
        INIT

        :param int d_model: Size of the input feature vectors
        :param int nhead: Number of heads in the multi-head attention
        :param int num_layers: number of transformer layers in the encoder
        :param float dropout: dropout probability of transformer layers in the encoder
        :param int sequence_length: Length of the input sequence

        """
        super(EncoderTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length

        cls_token = torch.zeros(1, 1, self.d_model).float()
        cls_token.require_grad = True
        self.clsToken = torch.nn.Parameter(cls_token)
        torch.nn.init.normal_(cls_token, std=0.02)

        self.pe = PositionalEmbedding(
            sequence_length=self.sequence_length + 1, d_model=self.d_model
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        for parameter in self.transformer_encoder.parameters():
            parameter.requires_grad = True

    def forward(
        self, features: torch.tensor, attention_mask: torch.tensor = None
    ) -> torch.tensor:
        """
        Forward pass

        :param torch.tensor features: Input features [batch_size, sequence_length, embedded_size]
        :param torch.tensor attention_mask: Mask for the input features
                                            [batch_size*heads, sequence_length, sequence_length]
                                            1 for masked positions and 0 for unmasked positions
        :return: Output features [batch_size, d_model]
        """

        features = torch.cat(
            (self.clsToken.repeat(features.size(0), 1, 1), features), dim=1
        )
        features = self.pe(features)
        features = self.transformer_encoder(features, attention_mask)
        return features


class OutputLayer(nn.Module):
    """
    Output layer of the model
    Based on RobertaClassificationHead:
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/modeling_roberta.py
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout_encoder_features: float = 0.2,
        from_transformer: bool = True,
    ):
        """
        INIT

        :param int d_model: Size of the encoder output vector
        :param int num_classes: Size of output vector
        :param float dropout_encoder_features: Dropout probability of the encoder output
        :param bool from_transformer: If true, get the CLS token from the transformer output
        """
        super(OutputLayer, self).__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.dropout_encoder_features = dropout_encoder_features
        self.dense = nn.Linear(self.d_model, self.d_model)
        self.dp = nn.Dropout(p=dropout_encoder_features)
        self.out_proj = nn.Linear(self.d_model, self.num_classes)
        self.tanh = nn.Tanh()
        self.from_transformer = from_transformer

    def forward(self, x):
        """
        Forward pass

        :param torch.tensor x: Input features [batch_size, d_model] if RNN else [batch_size, sequence_length+1, d_model]
        :return: Output features [num_classes]
        """
        if self.from_transformer:
            x = x[:, 0, :]  # Get [CLS] token
        x = self.dp(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dp(x)
        x = self.out_proj(x)
        return x


class OutputImageOrderingLayer(nn.Module):
    """
    Output layer of the image reordering model
    Based on  RobertaLMHead:
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/modeling_roberta.py
    """

    def __init__(self, d_model: int, num_classes: int):
        """
        INIT

        :param int d_model: Size of the encoder output vector
        :param int num_classes: Size of output vector
        """

        super(OutputImageOrderingLayer, self).__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.dense = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-05)
        self.decoder = nn.Linear(self.d_model, self.num_classes)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.decoder.bias = self.bias
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        Forward pass

        :param torch.tensor x: Input features [batch_size, sequence_length+1, d_model]
        :return: Output features [num_classes]
        """
        x = self.dense(x)[:, 1:, :]  # Remove CLS
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class Controller2Keyboard(nn.Module):
    """
    Map controller output to keyboard keys probabilities
    """

    def __init__(self):
        """
        INIT
        """
        super(Controller2Keyboard, self).__init__()
        keys2vector_matrix = torch.tensor(
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            requires_grad=False,
        )

        self.register_buffer("keys2vector_matrix", keys2vector_matrix)

    def forward(self, x: torch.tensor):
        """
        Forward pass

        :param torch.tensor x: Controller input [2]
        :return: Keyboard keys probabilities [9]
        """
        return 1.0 / torch.cdist(x, self.keys2vector_matrix)


class Keyboard2Controller(nn.Module):
    """
    Map keyboard keys probabilities to controller output
    """

    def __init__(self):
        """
        INIT
        """
        super(Keyboard2Controller, self).__init__()
        keys2vector_matrix = torch.tensor(
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            requires_grad=False,
        )

        self.register_buffer("keys2vector_matrix", keys2vector_matrix)

    def forward(self, x: torch.tensor):
        """
        Forward pass

        :param torch.tensor x: Keyboard keys probabilities [9]
        :return: Controller input [2]
        """
        controller_inputs = self.keys2vector_matrix.repeat(len(x), 1, 1)
        return (
            torch.sum(controller_inputs * x.view(len(x), 9, 1), dim=1)
            / torch.sum(x, dim=-1)[:, None]
        )


class TEDD1104LSTM(nn.Module):
    """
    T.E.D.D 1104 model with LSTM encoder. The model consists of:
         - A CNN that extract features from the images
         - A RNN (LSTM) that extracts a representation of the image sequence
         - A linear output layer that predicts the controller input.
    """

    def __init__(
        self,
        cnn_model_name: str,
        pretrained_cnn: bool,
        embedded_size: int,
        hidden_size: int,
        num_layers_lstm: int,
        bidirectional_lstm: bool,
        dropout_cnn_out: float,
        dropout_lstm: float,
        dropout_encoder_features: float,
        control_mode: str = "keyboard",
        sequence_size: int = 5,
    ):
        """
        INIT

        :param int embedded_size: Size of the output embedding
        :param float dropout_cnn_out: Dropout rate for the output of the CNN
        :param str cnn_model_name: Name of the CNN model from torchvision.models
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights
        :param int embedded_size: Size of the input feature vectors
        :param int hidden_size: LSTM hidden size
        :param int num_layers_lstm: number of layers in the LSTM
        :param bool bidirectional_lstm: forward or bidirectional LSTM
        :param float dropout_lstm: dropout probability for the LSTM
        :param float dropout_encoder_features: Dropout probability of the encoder output
        :param int sequence_size: Length of the input sequence
        :param control_mode: Model output format: keyboard (Classification task: 9 classes) or controller (Regression task: 2 variables)
        """

        super(TEDD1104LSTM, self).__init__()

        # Remember hyperparameters.
        self.cnn_model_name: str = cnn_model_name
        self.pretrained_cnn: bool = pretrained_cnn
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.hidden_size: int = hidden_size
        self.num_layers_lstm: int = num_layers_lstm
        self.bidirectional_lstm: bool = bidirectional_lstm
        self.dropout_cnn_out: float = dropout_cnn_out
        self.dropout_lstm: float = dropout_lstm
        self.dropout_encoder_features = dropout_encoder_features
        self.control_mode = control_mode

        self.EncoderCNN: EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,
            dropout_cnn_out=dropout_cnn_out,
            cnn_model_name=cnn_model_name,
            pretrained_cnn=pretrained_cnn,
        )

        self.EncoderRNN: EncoderRNN = EncoderRNN(
            embedded_size=embedded_size,
            hidden_size=hidden_size,
            num_layers=num_layers_lstm,
            bidirectional_lstm=bidirectional_lstm,
            dropout_lstm=dropout_lstm,
        )

        self.OutputLayer: OutputLayer = OutputLayer(
            d_model=embedded_size if not self.bidirectional_lstm else embedded_size * 2,
            num_classes=9 if self.control_mode == "keyboard" else 2,
            dropout_encoder_features=self.dropout_encoder_features,
            from_transformer=False,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        :param torch.tensor x: Input tensor of shape [batch_size * sequence_size, 3, 270, 480]
        :return: Output tensor of shape [9] if control_mode == "keyboard" or [2] if control_mode == "controller"
        """
        x = self.EncoderCNN(x)
        x = self.EncoderRNN(x)
        return self.OutputLayer(x)


class TEDD1104Transformer(nn.Module):
    """
    T.E.D.D 1104 model with transformer encoder. The model consists of:
         - A CNN that extract features from the images
         - A transformer that extracts a representation of the image sequence
         - A linear output layer that predicts the controller input.
    """

    def __init__(
        self,
        cnn_model_name: str,
        pretrained_cnn: bool,
        embedded_size: int,
        nhead: int,
        num_layers_transformer: int,
        dropout_cnn_out: float,
        positional_embeddings_dropout: float,
        dropout_transformer: float,
        dropout_encoder_features: float,
        control_mode: str = "keyboard",
        sequence_size: int = 5,
    ):
        """
        INIT

        :param int embedded_size: Size of the output embedding
        :param float dropout_cnn_out: Dropout rate for the output of the CNN
        :param str cnn_model_name: Name of the CNN model from torchvision.models
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights
        :param int embedded_size: Size of the input feature vectors
        :param int nhead: Number of heads in the multi-head attention
        :param int num_layers_transformer: number of transformer layers in the encoder
        :param float positional_embeddings_dropout: Dropout rate for the positional embeddings
        :param float dropout_transformer: dropout probability of transformer layers in the encoder
        :param int sequence_size: Length of the input sequence
        :param float dropout_encoder_features: Dropout probability of the encoder output
        :param control_mode: Model output format: keyboard (Classification task: 9 classes) or controller (Regression task: 2 variables)
        """
        super(TEDD1104Transformer, self).__init__()

        # Remember hyperparameters.
        self.cnn_model_name: str = cnn_model_name
        self.pretrained_cnn: bool = pretrained_cnn
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.nhead: int = nhead
        self.num_layers_transformer: int = num_layers_transformer
        self.dropout_cnn_out: float = dropout_cnn_out
        self.positional_embeddings_dropout: float = positional_embeddings_dropout
        self.dropout_transformer: float = dropout_transformer
        self.control_mode = control_mode
        self.dropout_encoder_features = dropout_encoder_features

        self.EncoderCNN: EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,
            dropout_cnn_out=dropout_cnn_out,
            cnn_model_name=cnn_model_name,
            pretrained_cnn=pretrained_cnn,
            sequence_size=self.sequence_size,
        )

        self.PositionalEncoding = PositionalEmbedding(
            d_model=embedded_size,
            dropout=self.positional_embeddings_dropout,
            sequence_length=self.sequence_size,
        )

        self.EncoderTransformer: EncoderTransformer = EncoderTransformer(
            d_model=embedded_size,
            nhead=nhead,
            num_layers=num_layers_transformer,
            dropout=self.dropout_transformer,
        )

        self.OutputLayer: OutputLayer = OutputLayer(
            d_model=embedded_size,
            num_classes=9 if self.control_mode == "keyboard" else 2,
            dropout_encoder_features=dropout_encoder_features,
            from_transformer=True,
        )

    def forward(
        self, x: torch.tensor, attention_mask: torch.tensor = None
    ) -> torch.tensor:
        """
        Forward pass

        :param torch.tensor x: Input tensor of shape [batch_size * sequence_size, 3, 270, 480]
        :param torch.tensor attention_mask: Mask for the input features
                                            [batch_size*heads, sequence_length, sequence_length]
                                            1 for masked positions and 0 for unmasked positions
        :return: Output tensor of shape [9] if control_mode == "keyboard" or [2] if control_mode == "controller"
        """
        x = self.EncoderCNN(x)
        x = self.PositionalEncoding(x)
        x = self.EncoderTransformer(x, attention_mask=attention_mask)
        return self.OutputLayer(x)


class TEDD1104TransformerForImageReordering(nn.Module):
    """
    T.E.D.D 1104 for image reordering model consists of:
         - A CNN that extract features from the images
         - A transformer that extracts a representation of the image sequence
         - A linear output layer that predicts the correct order of the input sequence
    """

    def __init__(
        self,
        cnn_model_name: str,
        pretrained_cnn: bool,
        embedded_size: int,
        nhead: int,
        num_layers_transformer: int,
        dropout_cnn_out: float,
        positional_embeddings_dropout: float,
        dropout_transformer: float,
        dropout_encoder_features: float,
        sequence_size: int = 5,
    ):
        """
        INIT

        :param int embedded_size: Size of the output embedding
        :param float dropout_cnn_out: Dropout rate for the output of the CNN
        :param str cnn_model_name: Name of the CNN model from torchvision.models
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights
        :param int embedded_size: Size of the input feature vectors
        :param int nhead: Number of heads in the multi-head attention
        :param int num_layers_transformer: number of transformer layers in the encoder
        :param float positional_embeddings_dropout: Dropout rate for the positional embeddings
        :param float dropout_transformer: dropout probability of transformer layers in the encoder
        :param int sequence_size: Length of the input sequence
        :param float dropout_encoder_features: Dropout probability of the encoder output
        :param sequence_size: Length of the input sequence
        """

        super(TEDD1104TransformerForImageReordering, self).__init__()

        # Remember hyperparameters.
        self.cnn_model_name: str = cnn_model_name
        self.pretrained_cnn: bool = pretrained_cnn
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.nhead: int = nhead
        self.num_layers_transformer: int = num_layers_transformer
        self.dropout_cnn_out: float = dropout_cnn_out
        self.positional_embeddings_dropout: float = positional_embeddings_dropout
        self.dropout_transformer: float = dropout_transformer
        self.dropout_encoder_features = dropout_encoder_features

        self.EncoderCNN: EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,
            dropout_cnn_out=dropout_cnn_out,
            cnn_model_name=cnn_model_name,
            pretrained_cnn=pretrained_cnn,
            sequence_size=self.sequence_size,
        )

        self.PositionalEncoding = PositionalEmbedding(
            d_model=embedded_size,
            dropout=self.positional_embeddings_dropout,
            sequence_length=self.sequence_size,
        )

        self.EncoderTransformer: EncoderTransformer = EncoderTransformer(
            d_model=embedded_size,
            nhead=nhead,
            num_layers=num_layers_transformer,
            dropout=self.dropout_transformer,
        )

        self.OutputLayer: OutputImageOrderingLayer = OutputImageOrderingLayer(
            d_model=embedded_size,
            num_classes=self.sequence_size,
        )

    def forward(
        self, x: torch.tensor, attention_mask: torch.tensor = None
    ) -> torch.tensor:
        """
        Forward pass

        :param torch.tensor x: Input tensor of shape [batch_size * sequence_size, 3, 270, 480]
        :param torch.tensor attention_mask: Mask for the input features
                                            [batch_size*heads, sequence_length, sequence_length]
                                            1 for masked positions and 0 for unmasked positions
        :return: Output tensor of shape [9] if control_mode == "keyboard" or [2] if control_mode == "controller"
        """
        x = self.EncoderCNN(x)
        x = self.PositionalEncoding(x)
        x = self.EncoderTransformer(x, attention_mask=attention_mask)
        return self.OutputLayer(x)


class Tedd1104ModelPL(pl.LightningModule):
    """
    Pytorch Lightning module for the Tedd1104Model
    """

    def __init__(
        self,
        cnn_model_name: str,
        pretrained_cnn: bool,
        embedded_size: int,
        nhead: int,
        num_layers_encoder: int,
        lstm_hidden_size: int,
        dropout_cnn_out: float,
        positional_embeddings_dropout: float,
        dropout_encoder: float,
        dropout_encoder_features: float = 0.8,
        control_mode: str = "keyboard",
        sequence_size: int = 5,
        encoder_type: str = "transformer",
        bidirectional_lstm=True,
        weights: List[float] = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-3,
        label_smoothing: float = 0.0,
        accelerator: str = None,
    ):
        """
        INIT

        :param int embedded_size: Size of the output embedding
        :param float dropout_cnn_out: Dropout rate for the output of the CNN
        :param str cnn_model_name: Name of the CNN model from torchvision.models
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights
        :param int embedded_size: Size of the input feature vectors
        :param int nhead: Number of heads in the multi-head attention
        :param int num_layers_encoder: number of transformer layers in the encoder
        :param float positional_embeddings_dropout: Dropout rate for the positional embeddings
        :param int sequence_size: Length of the input sequence
        :param float dropout_encoder: Dropout rate for the encoder
        :param float dropout_encoder_features: Dropout probability of the encoder output
        :param int lstm_hidden_size: LSTM hidden size
        :param bool bidirectional_lstm: forward or bidirectional LSTM
        :param List[float] weights: List of weights for the loss function [9] if control_mode == "keyboard" or [2] if control_mode == "controller"
        :param float learning_rate: Learning rate
        :param float weight_decay: Weight decay
        :param str control_mode: Model output format: keyboard (Classification task: 9 classes) or controller (Regression task: 2 variables)
        :param str encoder_type: Encoder type: transformer or lstm
        :param float label_smoothing: Label smoothing for the classification task
        """

        super(Tedd1104ModelPL, self).__init__()

        self.encoder_type = encoder_type.lower()
        assert self.encoder_type in [
            "lstm",
            "transformer",
        ], f"Encoder type {self.encoder_type} not supported, supported feature encoders [lstm,transformer]."

        self.control_mode = control_mode.lower()

        assert self.control_mode in [
            "keyboard",
            "controller",
        ], f"{self.control_mode} control mode not supported. Supported dataset types: [keyboard, controller].  "

        self.cnn_model_name: str = cnn_model_name
        self.pretrained_cnn: bool = pretrained_cnn
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.nhead: int = nhead
        self.num_layers_encoder: int = num_layers_encoder
        self.dropout_cnn_out: float = dropout_cnn_out
        self.positional_embeddings_dropout: float = positional_embeddings_dropout
        self.dropout_encoder: float = dropout_encoder
        self.dropout_encoder_features = dropout_encoder_features
        self.bidirectional_lstm = bidirectional_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.weights = weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.accelerator = accelerator

        if self.encoder_type == "transformer":
            self.model = TEDD1104Transformer(
                cnn_model_name=self.cnn_model_name,
                pretrained_cnn=self.pretrained_cnn,
                embedded_size=self.embedded_size,
                nhead=self.nhead,
                num_layers_transformer=self.num_layers_encoder,
                dropout_cnn_out=self.dropout_cnn_out,
                positional_embeddings_dropout=self.positional_embeddings_dropout,
                dropout_transformer=self.dropout_encoder,
                control_mode=self.control_mode,
                sequence_size=self.sequence_size,
                dropout_encoder_features=self.dropout_encoder_features,
            )
        else:
            self.model = TEDD1104LSTM(
                cnn_model_name=self.cnn_model_name,
                pretrained_cnn=self.pretrained_cnn,
                embedded_size=self.embedded_size,
                hidden_size=self.lstm_hidden_size,
                num_layers_lstm=self.num_layers_encoder,
                bidirectional_lstm=self.bidirectional_lstm,
                dropout_cnn_out=self.dropout_cnn_out,
                dropout_lstm=self.dropout_encoder,
                control_mode=self.control_mode,
                sequence_size=self.sequence_size,
                dropout_encoder_features=self.dropout_encoder_features,
            )

        self.total_batches = 0
        self.running_loss = 0

        self.train_accuracy = torchmetrics.Accuracy(
            num_classes=9, top_k=1, average="macro"
        )

        self.test_accuracy_k1_macro = torchmetrics.Accuracy(
            num_classes=9, top_k=1, average="macro"
        )

        self.test_accuracy_k3_micro = torchmetrics.Accuracy(
            num_classes=9, top_k=1, average="micro"
        )

        self.validation_accuracy_k1_micro = torchmetrics.Accuracy(
            num_classes=9, top_k=1, average="micro"
        )
        self.validation_accuracy_k3_micro = torchmetrics.Accuracy(
            num_classes=9, top_k=3, average="micro"
        )
        self.validation_accuracy_k1_macro = torchmetrics.Accuracy(
            num_classes=9, top_k=1, average="macro"
        )
        self.validation_accuracy_k3_macro = torchmetrics.Accuracy(
            num_classes=9, top_k=3, average="macro"
        )

        self.test_accuracy_k1_micro = torchmetrics.Accuracy(
            num_classes=9, top_k=1, average="micro"
        )
        self.test_accuracy_k3_micro = torchmetrics.Accuracy(
            num_classes=9, top_k=3, average="micro"
        )
        self.test_accuracy_k1_macro = torchmetrics.Accuracy(
            num_classes=9, top_k=1, average="macro"
        )
        self.test_accuracy_k3_macro = torchmetrics.Accuracy(
            num_classes=9, top_k=3, average="macro"
        )

        if self.control_mode == "keyboard":
            self.criterion = CrossEntropyLoss(
                weights=self.weights, label_smoothing=self.label_smoothing
            )
            self.Keyboard2Controller = Keyboard2Controller()
        else:
            self.validation_distance = torchmetrics.MeanSquaredError()
            self.criterion = WeightedMseLoss(weights=self.weights)
            self.Controller2Keyboard = Controller2Keyboard()

        self.save_hyperparameters()

    def forward(self, x, output_mode: str = "keyboard", return_best: bool = True):
        """
        Forward pass of the model.

        :param x: input data [batch_size * sequence_size, 3, 270, 480]
        :param output_mode: output mode, either "keyboard" or "controller". If the model uses another mode, we will convert the output to the desired mode.
        :param return_best: if True, we will return the class probabilities, else we will return the class with the highest probability (only for "keyboard" output_mode)
        """
        x = self.model(x)
        if self.control_mode == "keyboard":
            x = torch.functional.F.softmax(x, dim=1)
            if output_mode == "keyboard":
                if return_best:
                    return torch.argmax(x, dim=1)
                else:
                    return x

            elif output_mode == "controller":
                return self.Keyboard2Controller(x)
            else:
                raise ValueError(
                    f"Output mode: {output_mode} not supported. Supported modes: [keyboard,controller]"
                )

        elif self.control_mode == "controller":
            if output_mode == "controller":
                return x
            elif output_mode == "keyboard":
                if return_best:
                    return self.argmax(self.Controller2Keyboard(x), dim=-1)
                else:
                    return self.Controller2Keyboard(x)
            else:
                raise ValueError(
                    f"Output mode: {output_mode} not supported. Supported modes: [keyboard,controller]"
                )

        else:
            raise ValueError(
                f"Control mode: {self.control_mode} not supported. Supported modes: [keyboard,controller]"
            )

    def training_step(self, batch, batch_idx):
        """
        Training step.

        :param batch: batch of data
        :param batch_idx: batch index
        """
        x, attention_mask, y = batch["images"], batch["attention_mask"], batch["y"]
        # x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self.total_batches += 1
        if self.accelerator != "tpu":
            self.running_loss += loss.item()
            self.log("Train/loss", loss, sync_dist=True)
            self.log(
                "Train/running_loss",
                self.running_loss / self.total_batches,
                sync_dist=True,
            )
        else:
            if self.total_batches % 200 == 0:
                self.log("Train/loss", loss, sync_dist=True)

        return (
            {"preds": preds.detach(), "y": y, "loss": loss}
            if self.accelerator != "tpu"
            else {"loss": loss}
        )

    def training_step_end(self, outputs):
        """
        Training step end.

        :param outputs: outputs of the training step
        """
        if self.accelerator != "tpu" and self.control_mode == "keyboard":
            self.train_accuracy(outputs["preds"], outputs["y"])
            self.log(
                "Train/acc_k@1_macro",
                self.train_accuracy,
            )

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        :param batch: batch of data
        :param batch_idx: batch index
        """
        x, y = batch["images"], batch["y"]
        # x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.forward(x, output_mode="keyboard", return_best=False)

        return {"preds": preds, "y": y}  # "loss":loss}

    def validation_step_end(self, outputs):
        """
        Validation step end.

        :param outputs: outputs of the validation step
        """
        self.validation_accuracy_k1_micro(outputs["preds"], outputs["y"])
        self.validation_accuracy_k3_micro(outputs["preds"], outputs["y"])
        self.validation_accuracy_k1_macro(outputs["preds"], outputs["y"])
        self.validation_accuracy_k3_macro(outputs["preds"], outputs["y"])

        self.log(
            "Validation/acc_k@1_micro",
            self.validation_accuracy_k1_micro,
        )
        self.log(
            "Validation/acc_k@3_micro",
            self.validation_accuracy_k3_micro,
        )

        self.log(
            "Validation/acc_k@1_macro",
            self.validation_accuracy_k1_macro,
        )
        self.log(
            "Validation/acc_k@3_macro",
            self.validation_accuracy_k3_macro,
        )

    def test_step(self, batch, batch_idx, dataset_idx: int = 0):
        """
        Test step.

        :param batch: batch of data
        :param batch_idx: batch index
        """
        x, y = batch["images"], batch["y"]
        # x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.forward(x, output_mode="keyboard", return_best=False)

        return {"preds": preds, "y": y}  # "loss":loss}

    def test_step_end(self, outputs):
        """
        Test step end.

        :param outputs: outputs of the test step
        """
        self.test_accuracy_k1_micro(outputs["preds"], outputs["y"])
        self.test_accuracy_k3_micro(outputs["preds"], outputs["y"])
        self.test_accuracy_k1_macro(outputs["preds"], outputs["y"])
        self.test_accuracy_k3_macro(outputs["preds"], outputs["y"])

        self.log(
            "Test/acc_k@1_micro",
            self.test_accuracy_k1_micro,
        )
        self.log(
            "Test/acc_k@3_micro",
            self.test_accuracy_k3_micro,
        )

        self.log(
            "Test/acc_k@1_macro",
            self.test_accuracy_k1_macro,
        )
        self.log(
            "Test/acc_k@3_macro",
            self.test_accuracy_k3_macro,
        )

    def configure_optimizers(self):
        """
        Configure optimizers.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", patience=2, verbose=True
                ),
                "monitor": "Validation/acc_k@1_macro",
            },
        }


class Tedd1104ModelPLForImageReordering(pl.LightningModule):
    """
    Pytorch Lightning module for the Tedd1104ModelForImageReordering model
    """

    def __init__(
        self,
        cnn_model_name: str,
        pretrained_cnn: bool,
        embedded_size: int,
        nhead: int,
        num_layers_encoder: int,
        dropout_cnn_out: float,
        positional_embeddings_dropout: float,
        dropout_encoder: float,
        dropout_encoder_features: float = 0.8,
        sequence_size: int = 5,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-3,
        encoder_type: str = "transformer",
        accelerator: str = None,
    ):

        """
        INIT

        :param int embedded_size: Size of the output embedding
        :param float dropout_cnn_out: Dropout rate for the output of the CNN
        :param str cnn_model_name: Name of the CNN model from torchvision.models
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights
        :param int embedded_size: Size of the input feature vectors
        :param int nhead: Number of heads in the multi-head attention
        :param int num_layers_encoder: number of transformer layers in the encoder
        :param float positional_embeddings_dropout: Dropout rate for the positional embeddings
        :param int sequence_size: Length of the input sequence
        :param float dropout_encoder: Dropout rate for the encoder
        :param float dropout_encoder_features: Dropout probability of the encoder output
        :param float learning_rate: Learning rate
        :param float weight_decay: Weight decay
        :param str encoder_type: Encoder type: transformer or lstm
        """

        super(Tedd1104ModelPLForImageReordering, self).__init__()
        assert encoder_type == "transformer", "Only transformer encoder is supported"
        self.cnn_model_name: str = cnn_model_name
        self.pretrained_cnn: bool = pretrained_cnn
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.nhead: int = nhead
        self.num_layers_encoder: int = num_layers_encoder
        self.dropout_cnn_out: float = dropout_cnn_out
        self.positional_embeddings_dropout: float = positional_embeddings_dropout
        self.dropout_encoder: float = dropout_encoder
        self.dropout_encoder_features = dropout_encoder_features
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.encoder_type = encoder_type
        self.accelerator = accelerator

        self.model = TEDD1104TransformerForImageReordering(
            cnn_model_name=self.cnn_model_name,
            pretrained_cnn=self.pretrained_cnn,
            embedded_size=self.embedded_size,
            nhead=self.nhead,
            num_layers_transformer=self.num_layers_encoder,
            dropout_cnn_out=self.dropout_cnn_out,
            positional_embeddings_dropout=self.positional_embeddings_dropout,
            dropout_transformer=self.dropout_encoder,
            sequence_size=self.sequence_size,
            dropout_encoder_features=self.dropout_encoder_features,
        )

        self.total_batches = 0
        self.running_loss = 0

        self.train_accuracy = ImageReorderingAccuracy()
        self.validation_accuracy = ImageReorderingAccuracy()
        self.test_accuracy = ImageReorderingAccuracy()

        self.criterion = CrossEntropyLossImageReorder()

        self.save_hyperparameters()

    def forward(self, x, return_best: bool = True):
        """
        Forward pass of the model.

        :param x: input data [batch_size * sequence_size, 3, 270, 480]
        :param return_best: if True, we will return the class probabilities, else we will return the class with the highest probability
        """
        x = self.model(x)
        if return_best:
            return torch.argmax(x, dim=-1)
        else:
            return x

    def training_step(self, batch, batch_idx):
        """
        Training step.

        :param batch: batch of data
        :param batch_idx: batch index
        """
        x, y = batch["images"], batch["y"]
        # x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self.total_batches += 1

        if self.accelerator != "tpu":
            self.running_loss += loss.item()
            self.log("Train/loss", loss, sync_dist=True)
            self.log(
                "Train/running_loss",
                self.running_loss / self.total_batches,
                sync_dist=True,
            )
        else:
            if self.total_batches % 200 == 0:
                self.log("Train/loss", loss, sync_dist=True)

        return (
            {"preds": torch.argmax(preds.detach(), dim=-1), "y": y, "loss": loss}
            if self.accelerator != "tpu"
            else {"loss": loss}
        )

    def training_step_end(self, outputs):
        """
        Training step end.

        :param outputs: outputs of the training step
        """
        if self.accelerator != "tpu":
            self.train_accuracy(outputs["preds"], outputs["y"])
            self.log(
                "Train/acc",
                self.train_accuracy,
            )

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        :param batch: batch of data
        :param batch_idx: batch index
        """
        x, y = batch["images"], batch["y"]
        # x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.forward(x, return_best=True)

        return {"preds": preds, "y": y}

    def validation_step_end(self, outputs):
        """
        Validation step end.

        :param outputs: outputs of the validation step
        """
        self.validation_accuracy(outputs["preds"], outputs["y"])
        self.log(
            "Validation/acc",
            self.validation_accuracy,
        )

    def test_step(self, batch, batch_idx, dataset_idx: int = 0):
        """
        Test step.

        :param batch: batch of data
        :param batch_idx: batch index
        """
        x, y = batch["images"], batch["y"]
        # x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.forward(x, return_best=True)

        return {"preds": preds, "y": y}

    def test_step_end(self, outputs):
        """
        Test step end.

        :param outputs: outputs of the test step
        """
        self.test_accuracy(outputs["preds"], outputs["y"])

        self.log(
            "Test/acc",
            self.test_accuracy,
        )

    def configure_optimizers(self):
        """
        Configure optimizers.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", patience=2, verbose=True
                ),
                "monitor": "Validation/acc",
            },
        }
