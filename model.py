from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet
import pytorch_lightning as pl
import torchmetrics


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
        Input:
        - predicted: torch.tensor [batch_size, 3] Output from the cnn_model_name
        - predicted: torch.tensor [batch_size, 3] Gold values


        Output:
        -weighted_mse_loss columwise: torch.tensor  [1] if reduction == "mean"
                                                    [2] if reduction == "sum"
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
    ):
        """
        INIT
        Input:
        - weights:  torch.tensor [9] Weights for each variable
        - reduction:  reduction method: sum or mean

        """

        assert reduction in ["sum", "mean"], (
            f"Reduction method: {reduction} not implemented. "
            f"Available reduction methods: [sum,mean]"
        )

        super(CrossEntropyLoss, self).__init__()

        self.reduction = reduction
        if not weights:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        weights = torch.tensor(weights)
        weights.requires_grad = False

        self.register_buffer("weights", weights)

        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss(
            reduction=reduction, weight=weights
        )

    def forward(
        self,
        predicted: torch.tensor,
        target: torch.tensor,
    ) -> torch.tensor:

        """
        Input:
        - predicted: torch.tensor [batch_size, 9] Output from the cnn_model_name
        - predicted: torch.tensor [batch_size] Gold values


        Output:
        -Weighted CrossEntropyLoss: torch.tensor  [1] if reduction == "mean"
                                                    [9] if reduction == "sum"
        """
        return self.CrossEntropyLoss(predicted.view(-1, 9), target.view(-1).long())


class ImageReorderingAccuracy(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        assert (
            preds.size() == target.size()
        ), f"Pred sise: {preds.size()} != Target size: {target.size()}"

        self.correct += torch.sum(torch.all(preds == target, dim=-1))
        self.total += target.size(0)

    def compute(self):
        return self.correct.float() / self.total


def get_cnn(cnn_model_name: str, pretrained: bool) -> (torchvision.models, int):
    """
    Get resnet cnn_model_name

    Output:
     torchvision.models.resnet[18,34,50,101,152]

    Hyperparameters:
    - cnn_model_name: Resnet cnn_model_name from torchvision.models (number of layers): [18,34,50,101,152]
    - pretrained: Load cnn_model_name pretrained weights
    """

    cnn_call_method = getattr(models, cnn_model_name)
    cnn_model = cnn_call_method(pretrained=pretrained)
    _ = cnn_model._modules.popitem()
    cnn_model = nn.Sequential(*list(cnn_model.children()))

    # Test output_size
    features = cnn_model(torch.zeros((1, 3, 270, 480), dtype=torch.float32))
    output_size: int = features.reshape(features.size(0), -1).size(1)

    return cnn_model, output_size


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
        dropout_cnn_out: float,
        cnn_model_name: str,
        pretrained_cnn: bool,
        sequence_size: int,
    ):
        super(EncoderCNN, self).__init__()

        self.embedded_size = embedded_size
        self.cnn_model_name = cnn_model_name
        self.dropout_cnn_out = dropout_cnn_out
        self.pretrained_cnn = pretrained_cnn
        self.sequence_size = sequence_size

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

    def forward(self, images: torch.tensor) -> torch.tensor:
        features = self.cnn(images)
        features = features.reshape(features.size(0), -1)

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
    ):
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
        output, (h_n, c_n) = self.lstm(features)
        if self.bidirectional_lstm:
            x = torch.cat((h_n[-2], h_n[-1]), 1)
        else:
            x = h_n[-1]
        return x


class PositionalEmbedding(nn.Module):
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

    def __init__(self, d_model: int, sequence_length: int = 5, dropout: float = 0.1):
        super(PositionalEmbedding, self).__init__()

        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = dropout

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(sequence_length, d_model).float()
        pe.requires_grad = True
        pe = pe.unsqueeze(0)
        self.pe = torch.nn.Parameter(pe)
        torch.nn.init.normal_(self.pe, std=0.02)
        self.LayerNorm = nn.LayerNorm(self.d_model, eps=1e-05)
        self.dp = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        pe = self.pe[:, : x.size(1)]
        x = pe + x
        x = self.LayerNorm(x)
        x = self.dp(x)
        return x


class EncoderTransformer(nn.Module):
    """
    Extract feature vectors from input images (Transformer Encoder)

    Input:
     torch.tensor [batch_size, sequence_size, embedded_size]

    Output:
     torch.tensor [batch_size, hidden_size]

     Hyperparameters:
    - d_model: Size of the input feature vectors
    - nhead: LSTM hidden size
    - num_layers_transformer: number of transformer layers in the encoder
    - dropout: dropout probability of transformer layers in the encoder
    - mask_prob: probability of masking each input vector
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 1,
        mask_prob: float = 0.2,
        dropout: float = 0.1,
        sequence_length: int = 5,
    ):
        super(EncoderTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.mask_prob = mask_prob
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
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        for parameter in self.transformer_encoder.parameters():
            parameter.requires_grad = True

    def forward(self, features: torch.tensor):
        if self.training:
            bernolli_matrix = (
                torch.cat(
                    (
                        torch.tensor([1]).float(),
                        (torch.tensor([self.mask_prob]).float()).repeat(
                            self.sequence_length
                        ),
                    ),
                    0,
                )
                .unsqueeze(0)
                .repeat([features.size(0) * self.nhead, 1])
            )
            bernolli_distributor = torch.distributions.Bernoulli(bernolli_matrix)
            sample = bernolli_distributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1)
        else:

            mask = torch.ones(
                features.size(0) * self.nhead,
                self.sequence_length + 1,
                self.sequence_length + 1,
            )

        mask = mask.type_as(features)
        features = torch.cat(
            (self.clsToken.repeat(features.size(0), 1, 1), features), dim=1
        )
        features = self.pe(features)
        # print(f"x.size(): {x.size()}. mask.size(): {mask.size()}")
        features = self.transformer_encoder(
            features.transpose(0, 1), mask=mask
        ).transpose(0, 1)

        return features


class OutputLayer(nn.Module):
    """

    FROM https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/modeling_roberta.py

    Output linear layer that produces the predictions

    Input:
     torch.tensor [batch_size, hidden_size]

    Output:
     Forward: torch.tensor [batch_size, num_classes] (output values without softmax)


    Hyperparameters:
    - d_model: Size of the feature vectors
    - num_classes: Number of classes, 9 for keyboard 3 for controller
    """

    def __init__(
        self, d_model: int, num_classes: int, dropout_encoder_features: float = 0.2
    ):
        super(OutputLayer, self).__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.dropout_encoder_features = dropout_encoder_features
        self.dense = nn.Linear(self.d_model, self.d_model)
        self.dp = nn.Dropout(p=dropout_encoder_features)
        self.out_proj = nn.Linear(self.d_model, self.num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x[:, 0, :]  # GET CLS TOKEN
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ImageOrderingLayer(nn.Module):
    """

    FROM https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/modeling_roberta.py

    Output linear layer that produces the predictions

    Input:
     torch.tensor [batch_size, hidden_size]

    Output:
     Forward: torch.tensor [batch_size, num_classes] (output values without softmax)


    Hyperparameters:
    - d_model: Size of the feature vectors
    - num_classes: Number of classes, 9 for keyboard 3 for controller
    """

    def __init__(
        self, d_model: int, num_classes: int, dropout_encoder_features: float = 0.2
    ):
        super(ImageOrderingLayer, self).__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.dense = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-05)
        self.decoder = nn.Linear(self.d_model, self.num_classes)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.decoder.bias = self.bias
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dense(x)[:, 1:, :]  # Remove CLS
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class Controller2Keyboard(nn.Module):
    def __init__(self):
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
        return 1.0 / (torch.cdist(x, self.keys2vector_matrix) + 1.0)


class Keyboard2Controller(nn.Module):
    def __init__(self):
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
        controller_inputs = self.keys2vector_matrix.repeat(len(x), 1, 1)
        return (
            torch.sum(controller_inputs * x.view(len(x), 9, 1), dim=1)
            / torch.sum(x, dim=-1)[:, None]
        )


class TEDD1104LSTM(nn.Module):
    """
    T.E.D.D 1104 LSTM consists of 3 modules:
        [*] A CNN (Resnet) that extract features from the images
        [*] A RNN (LSTM) that generates a representation of the sequence of features from the CNN
        [*] A linear output layer that predicts the controller input.

    Input:
     torch.tensor [batch_size, num_channels, H, W]
     For efficiency the input input is not packed as sequence of 5 images, all the images in the batch will be
     encoded in the CNN and the features vectors will be packed as sequences of 5 vectors before feeding them to the
     RNN.

    Output:
     Forward: torch.tensor [batch_size, 9 for keyboard 3 for controller] (output values without softmax)

    Hyperparameters:
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights
    - sequence_size: Length of each series of features
    - embedded_size: Size of the feature vectors
    - hidden_size: LSTM hidden size
    - num_layers_lstm: number of layers in the LSTM
    - bidirectional_lstm: forward or bidirectional LSTM
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_cnn_out: dropout probability for the cnn features (output layer)
    - dropout_lstm: dropout probability for the LSTM
    - dropout_encoder_features: Dropout probability for the LSTM output
    - control_mode: Keyboard: Classification cnn_model_name with 9 classes. Controller: Regression cnn_model_name with 3 variables

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
            sequence_size=self.sequence_size,
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
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.EncoderCNN(x)
        x = self.EncoderRNN(x)
        return self.OutputLayer(x)


class TEDD1104Transformer(nn.Module):
    """
    T.E.D.D 1104 Transformer consists of 3 modules:
        [*] A CNN (Resnet) that extract features from the images
        [*] A Transformer that generates a representation of the sequence of features from the CNN
        [*] A linear output layer that predicts the controller input.

    Input:
     torch.tensor [batch_size, num_channels, H, W]
     For efficiency the input input is not packed as sequence of 5 images, all the images in the batch will be
     encoded in the CNN and the features vectors will be packed as sequences of 5 vectors before feeding them to the
     RNN.

    Output:
     Forward: torch.tensor [batch_size, 9 for keyboard 3 for controller] (output values without softmax)


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
    - dropout_transformer: dropout probability for the transformer layers
    - dropout_encoder_features: Dropout probability for the transformer output
    - mask_prob: probability of masking each input vector before the transformer
    - control_mode: Keyboard: Classification cnn_model_name with 9 classes. Controller: Regression cnn_model_name with 3 variables

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
        mask_prob: float,
        control_mode: str = "keyboard",
        sequence_size: int = 5,
    ):
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
        self.mask_prob = mask_prob
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
            d_model=embedded_size, dropout=self.positional_embeddings_dropout
        )

        self.EncoderTransformer: EncoderTransformer = EncoderTransformer(
            d_model=embedded_size,
            nhead=nhead,
            num_layers=num_layers_transformer,
            mask_prob=self.mask_prob,
            dropout=self.dropout_transformer,
        )

        self.OutputLayer: OutputLayer = OutputLayer(
            d_model=embedded_size,
            num_classes=9 if self.control_mode == "keyboard" else 2,
            dropout_encoder_features=dropout_encoder_features,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.EncoderCNN(x)
        x = self.PositionalEncoding(x)
        x = self.EncoderTransformer(x)
        return self.OutputLayer(x)


class TEDD1104TransformerForImageReordering(nn.Module):
    """
    T.E.D.D 1104 Transformer consists of 3 modules:
        [*] A CNN (Resnet) that extract features from the images
        [*] A Transformer that generates a representation of the sequence of features from the CNN
        [*] A linear output layer that predicts the controller input.

    Input:
     torch.tensor [batch_size, num_channels, H, W]
     For efficiency the input input is not packed as sequence of 5 images, all the images in the batch will be
     encoded in the CNN and the features vectors will be packed as sequences of 5 vectors before feeding them to the
     RNN.

    Output:
     Forward: torch.tensor [batch_size, 9 for keyboard 3 for controller] (output values without softmax)


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
    - dropout_transformer: dropout probability for the transformer layers
    - dropout_encoder_features: Dropout probability for the transformer output
    - mask_prob: probability of masking each input vector before the transformer
    - control_mode: Keyboard: Classification cnn_model_name with 9 classes. Controller: Regression cnn_model_name with 3 variables

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
        mask_prob: float,
        sequence_size: int = 5,
    ):
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
        self.mask_prob = mask_prob
        self.dropout_encoder_features = dropout_encoder_features

        self.EncoderCNN: EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,
            dropout_cnn_out=dropout_cnn_out,
            cnn_model_name=cnn_model_name,
            pretrained_cnn=pretrained_cnn,
            sequence_size=self.sequence_size,
        )

        self.PositionalEncoding = PositionalEmbedding(
            d_model=embedded_size, dropout=self.positional_embeddings_dropout
        )

        self.EncoderTransformer: EncoderTransformer = EncoderTransformer(
            d_model=embedded_size,
            nhead=nhead,
            num_layers=num_layers_transformer,
            mask_prob=self.mask_prob,
            dropout=self.dropout_transformer,
        )

        self.OutputLayer: ImageOrderingLayer = ImageOrderingLayer(
            d_model=embedded_size,
            num_classes=self.sequence_size,
            dropout_encoder_features=dropout_encoder_features,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.EncoderCNN(x)
        x = self.PositionalEncoding(x)
        x = self.EncoderTransformer(x)
        return self.OutputLayer(x)


class Tedd1104ModelPL(pl.LightningModule):
    """
    T.E.D.D. 1104 (https://nazizombiesplus.fandom.com/wiki/T.E.D.D.) is the neural network that learns
    how to drive in videogames. It has been develop with Grand Theft Auto V (GTAV) in mind. However
    it can learn how to drive in any videogame and if the cnn_model_name and controls are modified accordingly
    it can play any game. The cnn_model_name receive as input 5 consecutive images that have been captured
    with a fixed time interval between then (by default 1/10 seconds) and learns the correct
    controller input.

    T.E.D.D 1104 consists of 3 modules:
        [*] A CNN (Resnet) that extract features from the images
        [*] A Transformer or LSTM that generates a representation of the sequence of features from the CNN
        [*] A linear output layer that predicts the controller input.

    Input:
     torch.tensor [batch_size, num_channels, H, W]
     For efficiency the input input is not packed as sequence of 5 images, all the images in the batch will be
     encoded in the CNN and the features vectors will be packed as sequences of 5 vectors before feeding them to the
     RNN.

    Output:
     Forward: torch.tensor [batch_size, 9 for keyboard 3 for controller] (output values without softmax)


    Hyperparameters:
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights
    - sequence_size: Length of each series of features
    - embedded_size: Size of the feature vectors
    - nhead: number of heads for the transformer layer
    - num_layers_transformer: number of transformer layers in the encoder
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_cnn_out: dropout probability for the cnn features (output layer)
    - positional_embeddings_dropout: dropout probability for the transformer input embeddings
    - dropout_transformer_out: dropout probability for the transformer features (output layer)
    - mask_prob: probability of masking each input vector before the transformer
    - control_mode: Keyboard: Classification cnn_model_name with 9 classes. Controller: Regression cnn_model_name with 3 variables
    - Encoder type: Use LSTM or Transformer as feature encoder [lstm, transformer]
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
        mask_prob: float,
        dropout_encoder_features: float = 0.8,
        control_mode: str = "keyboard",
        sequence_size: int = 5,
        encoder_type: str = "transformer",
        bidirectional_lstm=True,
        weights: List[float] = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-3,
    ):

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
        self.mask_prob = mask_prob
        self.bidirectional_lstm = bidirectional_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.weights = weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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
                mask_prob=self.mask_prob,
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

        self.validation_accuracy_k1 = torchmetrics.Accuracy(
            num_classes=9, top_k=1, average="micro"
        )
        self.validation_accuracy_k3 = torchmetrics.Accuracy(
            num_classes=9, top_k=3, average="micro"
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
            self.criterion = CrossEntropyLoss(weights=self.weights)
            self.Keyboard2Controller = Keyboard2Controller()
        else:
            self.validation_distance = torchmetrics.MeanSquaredError()
            self.criterion = WeightedMseLoss(weights=self.weights)
            self.Controller2Keyboard = Controller2Keyboard()

        self.save_hyperparameters()

    def forward(self, x, output_mode: str = "keyboard", return_best: bool = True):
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
        x, y = batch["images"], batch["y"]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self.total_batches += 1
        self.running_loss += loss.item()
        self.log("Train/loss", loss, sync_dist=True)
        self.log(
            "Train/running_loss", self.running_loss / self.total_batches, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["images"], batch["y"]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.forward(x, output_mode="keyboard", return_best=False)
        # loss = self.criterion(preds, y)
        # self.log("Val/loss", loss, sync_dist=True)

        # if self.control_mode == "keyboard":
        #    preds = torch.functional.F.softmax(preds, dim=1)

        return {"preds": preds, "y": y}  # "loss":loss}

    def validation_step_end(self, outputs):
        self.validation_accuracy_k1(outputs["preds"], outputs["y"])
        self.validation_accuracy_k3(outputs["preds"], outputs["y"])
        self.log(
            "Val/acc_k@1",
            self.validation_accuracy_k1,
        )

        self.log(
            "Val/acc_k@3",
            self.validation_accuracy_k3,
        )

        """
        if self.control_mode == "keyboard":
            self.validation_accuracy_k1(outputs["preds"], outputs["y"])
            self.validation_accuracy_k3(outputs["preds"], outputs["y"])
            self.log(
                "Val/acc_k@1",
                self.validation_accuracy_k1,
            )

            self.log(
                "Val/acc_k@3",
                self.validation_accuracy_k3,
            )
        else:
            self.validation_distance(outputs["preds"], outputs["y"])
            self.log(
                "Val/mse",
                self.validation_distance,
            )
        """

    def test_step(self, batch, batch_idx, dataset_idx: int = 0):
        x, y = batch["images"], batch["y"]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.forward(x, output_mode="keyboard", return_best=False)
        # loss = self.criterion(preds, y)
        # self.log("Val/loss", loss, sync_dist=True)

        # if self.control_mode == "keyboard":
        #    preds = torch.functional.F.softmax(preds, dim=1)

        return {"preds": preds, "y": y}  # "loss":loss}

    def test_step_end(self, outputs):
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

        """
        if self.control_mode == "keyboard":
            self.test_accuracy_k1(outputs["preds"], outputs["y"])
            self.test_accuracy_k3(outputs["preds"], outputs["y"])
            self.log(
                "Test/acc_k@1",
                self.test_accuracy_k1,
            )

            self.log(
                "Test/acc_k@3",
                self.test_accuracy_k3,
            )
        else:
            self.test_distance(outputs["preds"], outputs["y"])
            self.log(
                "Test/mse",
                self.test_distance,
            )
        """

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", patience=5, verbose=True
                ),
                "monitor": "Val/acc_k@1",
            },
        }


class Tedd1104ModelPLForImageReordering(pl.LightningModule):
    """
    T.E.D.D. 1104 (https://nazizombiesplus.fandom.com/wiki/T.E.D.D.) is the neural network that learns
    how to drive in videogames. It has been develop with Grand Theft Auto V (GTAV) in mind. However
    it can learn how to drive in any videogame and if the cnn_model_name and controls are modified accordingly
    it can play any game. The cnn_model_name receive as input 5 consecutive images that have been captured
    with a fixed time interval between then (by default 1/10 seconds) and learns the correct
    controller input.

    T.E.D.D 1104 consists of 3 modules:
        [*] A CNN (Resnet) that extract features from the images
        [*] A Transformer or LSTM that generates a representation of the sequence of features from the CNN
        [*] A linear output layer that predicts the controller input.

    Input:
     torch.tensor [batch_size, num_channels, H, W]
     For efficiency the input input is not packed as sequence of 5 images, all the images in the batch will be
     encoded in the CNN and the features vectors will be packed as sequences of 5 vectors before feeding them to the
     RNN.

    Output:
     Forward: torch.tensor [batch_size, 9 for keyboard 3 for controller] (output values without softmax)


    Hyperparameters:
    - resnet: resnet module to use [18,34,50,101,152]
    - pretrained_resnet: Load pretrained resnet weights
    - sequence_size: Length of each series of features
    - embedded_size: Size of the feature vectors
    - nhead: number of heads for the transformer layer
    - num_layers_transformer: number of transformer layers in the encoder
    - dropout_cnn: dropout probability for the CNN layers
    - dropout_cnn_out: dropout probability for the cnn features (output layer)
    - positional_embeddings_dropout: dropout probability for the transformer input embeddings
    - dropout_transformer_out: dropout probability for the transformer features (output layer)
    - mask_prob: probability of masking each input vector before the transformer
    - control_mode: Keyboard: Classification cnn_model_name with 9 classes. Controller: Regression cnn_model_name with 3 variables
    - Encoder type: Use LSTM or Transformer as feature encoder [lstm, transformer]
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
        mask_prob: float,
        dropout_encoder_features: float = 0.8,
        sequence_size: int = 5,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-3,
    ):

        super(Tedd1104ModelPLForImageReordering, self).__init__()

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
        self.mask_prob = mask_prob
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = TEDD1104TransformerForImageReordering(
            cnn_model_name=self.cnn_model_name,
            pretrained_cnn=self.pretrained_cnn,
            embedded_size=self.embedded_size,
            nhead=self.nhead,
            num_layers_transformer=self.num_layers_encoder,
            dropout_cnn_out=self.dropout_cnn_out,
            positional_embeddings_dropout=self.positional_embeddings_dropout,
            dropout_transformer=self.dropout_encoder,
            mask_prob=self.mask_prob,
            sequence_size=self.sequence_size,
            dropout_encoder_features=self.dropout_encoder_features,
        )

        self.total_batches = 0
        self.running_loss = 0

        self.train_accuracy = ImageReorderingAccuracy()
        self.validation_accuracy = ImageReorderingAccuracy()
        self.test_accuracy = ImageReorderingAccuracy()

        self.criterion = WeightedMseLoss()

        self.save_hyperparameters()

    def forward(self, x, return_best: bool = True):
        x = self.model(x)
        if return_best:
            return torch.argmax(x, dim=-1)
        else:
            return x

    def training_step(self, batch, batch_idx):
        x, y = batch["images"], batch["y"]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self.total_batches += 1
        self.running_loss += loss.item()
        self.log("Train/loss", loss, sync_dist=True)
        self.log(
            "Train/running_loss", self.running_loss / self.total_batches, sync_dist=True
        )
        return {"preds": torch.argmax(preds, dim=1), "y": y}

    def training_step_end(self, outputs):
        self.train_accuracy(outputs["preds"], outputs["y"])
        self.log(
            "Train/acc",
            self.validation_accuracy_k1,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch["images"], batch["y"]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.forward(x, return_best=True)

        return {"preds": preds, "y": y}

    def validation_step_end(self, outputs):
        self.validation_accuracy(outputs["preds"], outputs["y"])

        self.log(
            "Val/acc",
            self.validation_accuracy,
        )

    def test_step(self, batch, batch_idx, dataset_idx: int = 0):
        x, y = batch["images"], batch["y"]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.forward(x, return_best=True)

        return {"preds": preds, "y": y}

    def test_step_end(self, outputs):
        self.test_accuracy(outputs["preds"], outputs["y"])

        self.log(
            "Test/acc",
            self.test_accuracy_k1_micro,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", patience=5, verbose=True
                ),
                "monitor": "Val/acc_k@1",
            },
        }
