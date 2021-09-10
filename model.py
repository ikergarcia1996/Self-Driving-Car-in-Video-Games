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
            weights = [1.0, 1.0, 1.0]
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
        - predicted: torch.tensor [batch_size, 9] Output from the model
        - predicted: torch.tensor [batch_size] Gold values


        Output:
        -Weighted CrossEntropyLoss: torch.tensor  [1] if reduction == "mean"
                                                    [9] if reduction == "sum"
        """
        return self.CrossEntropyLoss(predicted.view(-1, 9), target.view(-1).long())


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
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(sequence_length, d_model).float()
        pe.requires_grad = True
        pe = pe.unsqueeze(0)
        self.pe = torch.nn.Parameter(pe)
        torch.nn.init.normal_(self.pe, std=0.02)
        self.dp = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        pe = self.pe[:, : x.size(1)]
        x = pe + x
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
    Output linear layer that produces the predictions

    Input:
     torch.tensor [batch_size, hidden_size]

    Output:
     Forward: torch.tensor [batch_size, num_classes] (output values without softmax)


    Hyperparameters:
    - d_model: Size of the feature vectors
    - num_classes: Number of classes, 9 for keyboard 3 for controller
    """

    def __init__(self, d_model: int, num_classes: int):
        super(OutputLayer, self).__init__()

        self.d_model = d_model
        self.num_classes = num_classes

        self.dp = torch.nn.Dropout(p=0.8)

        self.fc_action = torch.nn.Linear(self.d_model, self.num_classes)

        self.fc_action.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()

    def forward(self, x):
        self.dp(x)
        return self.fc_action(x)


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
    - control_mode: Keyboard: Classification model with 9 classes. Controller: Regression model with 3 variables

    """

    def __init__(
        self,
        resnet: int,
        pretrained_resnet: bool,
        embedded_size: int,
        hidden_size: int,
        num_layers_lstm: int,
        bidirectional_lstm: bool,
        dropout_cnn: float,
        dropout_cnn_out: float,
        dropout_lstm: float,
        control_mode: str = "keyboard",
        sequence_size: int = 5,
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
        self.dropout_cnn: float = dropout_cnn
        self.dropout_cnn_out: float = dropout_cnn_out
        self.dropout_lstm: float = dropout_lstm
        self.control_mode = control_mode

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
        )

        self.OutputLayer: OutputLayer = OutputLayer(
            d_model=embedded_size if not self.bidirectional_lstm else embedded_size * 2,
            num_classes=9 if self.control_mode == "keyboard" else 3,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.EncoderCNN(x)
        x = self.PackFeatureVectors(x)
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
    - dropout_transformer_out: dropout probability for the transformer features (output layer)
    - mask_prob: probability of masking each input vector before the transformer
    - control_mode: Keyboard: Classification model with 9 classes. Controller: Regression model with 3 variables

    """

    def __init__(
        self,
        resnet: int,
        pretrained_resnet: bool,
        embedded_size: int,
        nhead: int,
        num_layers_transformer: int,
        dropout_cnn: float,
        dropout_cnn_out: float,
        positional_embeddings_dropout: float,
        dropout_transformer: float,
        mask_prob: float,
        control_mode: str = "keyboard",
        sequence_size: int = 5,
    ):
        super(TEDD1104Transformer, self).__init__()

        # Remember hyperparameters.
        self.resnet: int = resnet
        self.pretrained_resnet: bool = pretrained_resnet
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.nhead: int = nhead
        self.num_layers_transformer: int = num_layers_transformer
        self.dropout_cnn: float = dropout_cnn
        self.dropout_cnn_out: float = dropout_cnn_out
        self.positional_embeddings_dropout: float = positional_embeddings_dropout
        self.dropout_transformer: float = dropout_transformer
        self.mask_prob = mask_prob
        self.control_mode = control_mode

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
            num_classes=9 if self.control_mode == "keyboard" else 3,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.EncoderCNN(x)
        x = self.PackFeatureVectors(x)
        x = self.PositionalEncoding(x)
        x = self.EncoderTransformer(x)[:, 0, :]
        return self.OutputLayer(x)


class Tedd1104ModelPL(pl.LightningModule):
    """
    T.E.D.D. 1104 (https://nazizombiesplus.fandom.com/wiki/T.E.D.D.) is the neural network that learns
    how to drive in videogames. It has been develop with Grand Theft Auto V (GTAV) in mind. However
    it can learn how to drive in any videogame and if the model and controls are modified accordingly
    it can play any game. The model receive as input 5 consecutive images that have been captured
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
    - control_mode: Keyboard: Classification model with 9 classes. Controller: Regression model with 3 variables
    - Encoder type: Use LSTM or Transformer as feature encoder [lstm, transformer]
    """

    def __init__(
        self,
        resnet: int,
        pretrained_resnet: bool,
        embedded_size: int,
        nhead: int,
        num_layers_encoder: int,
        lstm_hidden_size: int,
        dropout_cnn: float,
        dropout_cnn_out: float,
        positional_embeddings_dropout: float,
        dropout_encoder: float,
        mask_prob: float,
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

        self.resnet: int = resnet
        self.pretrained_resnet: bool = pretrained_resnet
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.nhead: int = nhead
        self.num_layers_encoder: int = num_layers_encoder
        self.dropout_cnn: float = dropout_cnn
        self.dropout_cnn_out: float = dropout_cnn_out
        self.positional_embeddings_dropout: float = positional_embeddings_dropout
        self.dropout_encoder: float = dropout_encoder
        self.mask_prob = mask_prob
        self.bidirectional_lstm = bidirectional_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.weights = weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if self.encoder_type == "transformer":
            self.model = TEDD1104Transformer(
                resnet=self.resnet,
                pretrained_resnet=self.pretrained_resnet,
                embedded_size=self.embedded_size,
                nhead=self.nhead,
                num_layers_transformer=self.num_layers_encoder,
                dropout_cnn=self.dropout_cnn,
                dropout_cnn_out=self.dropout_cnn_out,
                positional_embeddings_dropout=self.positional_embeddings_dropout,
                dropout_transformer=self.dropout_encoder,
                mask_prob=self.mask_prob,
                control_mode=self.control_mode,
                sequence_size=self.sequence_size,
            )
        else:
            self.model = TEDD1104LSTM(
                resnet=self.resnet,
                pretrained_resnet=self.pretrained_resnet,
                embedded_size=self.embedded_size,
                hidden_size=self.lstm_hidden_size,
                num_layers_lstm=self.num_layers_encoder,
                dropout_cnn=self.dropout_cnn,
                bidirectional_lstm=self.bidirectional_lstm,
                dropout_cnn_out=self.dropout_cnn_out,
                dropout_lstm=self.dropout_encoder,
                control_mode=self.control_mode,
                sequence_size=self.sequence_size,
            )

        self.total_batches = 0
        self.running_loss = 0

        if self.control_mode == "keyboard":
            self.validation_accuracy_k1 = torchmetrics.Accuracy(num_classes=9, top_k=1)
            self.validation_accuracy_k3 = torchmetrics.Accuracy(num_classes=9, top_k=3)
            self.test_accuracy_k1 = torchmetrics.Accuracy(num_classes=9, top_k=1)
            self.test_accuracy_k3 = torchmetrics.Accuracy(num_classes=9, top_k=3)

            self.criterion = CrossEntropyLoss(weights=self.weights)
        else:
            self.validation_distance = torchmetrics.MeanSquaredError()
            self.criterion = WeightedMseLoss(weights=self.weights)

    def forward(self, x):
        x = self.model(x)
        if self.control_mode == "keyboard":
            return torch.argmax(torch.functional.F.softmax(x, dim=2), dim=2)
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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["images"], batch["y"]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self.log("Val/loss", loss, sync_dist=True)

        if self.control_mode == "keyboard":
            preds = torch.functional.F.softmax(preds, dim=1)

        return {"loss": loss, "preds": preds, "y": y}

    def validation_step_end(self, outputs):
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

    def test_step(self, batch, batch_idx):
        x, y = batch["images"], batch["y"]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self.log("Test/loss", loss, sync_dist=True)
        if self.control_mode == "keyboard":
            preds = torch.functional.F.softmax(preds, dim=1)

        return {"loss": loss, "preds": preds, "y": y}

    def test_step_end(self, outputs):
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
                "monitor": "Val/loss",
            },
        }
