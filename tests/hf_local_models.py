"""
Small locally trained architectures for the parser robustness sweep.

These cover the categories missing from the Hugging Face hub (plain RNN,
LSTM, MLP, autoencoder, GAN, GNN). Each factory trains the model for one
step on a dummy batch so the weights and the running statistics are filled.
"""

# Import Modules
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core.model_schema import ModelBlockSchema  # noqa: F401 (type exports)


def _train_one_step(model: nn.Module, inputs: dict[str, Tensor]) -> None:
    """
    Train a model for one step on its dummy inputs.

    Args:
        model (nn.Module): The model to train, already in train mode.
        inputs (dict[str, Tensor]): The dummy inputs of the forward.
    """

    # One optimization step on the sum of the outputs as a dummy loss
    optimizer: torch.optim.Optimizer = torch.optim.SGD(
        params=model.parameters(), lr=0.01
    )
    optimizer.zero_grad()
    model(**inputs).sum().backward()
    optimizer.step()
    model.eval()


class MLPClassifier(nn.Module):
    """
    Plain multilayer perceptron classifier over flattened inputs.
    """

    def __init__(self) -> None:
        """Init function."""

        # Init parent
        super().__init__()

        # The layers of the perceptron
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward the inputs through the perceptron.

        Args:
            x (Tensor): The inputs of shape (batch, 784).

        Returns:
            Tensor: The logits of shape (batch, 10).
        """

        # The logits of the classifier
        return self.net(x)


class LSTMClassifier(nn.Module):
    """
    LSTM classifier over token ids.
    """

    def __init__(self) -> None:
        """Init function."""

        # Init parent
        super().__init__()

        # The token embedding, the recurrence and the classifier
        self.embedding: nn.Embedding = nn.Embedding(1000, 32)
        self.lstm: nn.LSTM = nn.LSTM(32, 64, batch_first=True)
        self.fc: nn.Linear = nn.Linear(64, 10)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Classify the token sequence.

        Args:
            input_ids (Tensor): The token ids of shape (batch, time).

        Returns:
            Tensor: The logits of shape (batch, 10).
        """

        # Embed then run the recurrence, and classify the last state
        embedded: Tensor = self.embedding(input_ids)
        outputs: Tensor
        hidden: tuple[Tensor, Tensor]
        outputs, hidden = self.lstm(embedded)
        return self.fc(hidden[0][-1])


class RNNClassifier(nn.Module):
    """
    Vanilla RNN classifier over token ids.
    """

    def __init__(self) -> None:
        """Init function."""

        # Init parent
        super().__init__()

        # The token embedding, the recurrence and the classifier
        self.embedding: nn.Embedding = nn.Embedding(1000, 32)
        self.rnn: nn.RNN = nn.RNN(32, 64, batch_first=True)
        self.fc: nn.Linear = nn.Linear(64, 10)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Classify the token sequence.

        Args:
            input_ids (Tensor): The token ids of shape (batch, time).

        Returns:
            Tensor: The logits of shape (batch, 10).
        """

        # Embed then run the recurrence, and classify the last state
        embedded: Tensor = self.embedding(input_ids)
        outputs: Tensor
        hidden: Tensor
        outputs, hidden = self.rnn(embedded)
        return self.fc(hidden[-1])


class GRUSeq2Seq(nn.Module):
    """
    GRU sequence to sequence logits over a vocabulary.
    """

    def __init__(self) -> None:
        """Init function."""

        # Init parent
        super().__init__()

        # The token embedding and the recurrence
        self.embedding: nn.Embedding = nn.Embedding(500, 32)
        self.gru: nn.GRU = nn.GRU(32, 64, batch_first=True)
        self.fc: nn.Linear = nn.Linear(64, 500)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Produce logits at every step.

        Args:
            input_ids (Tensor): The token ids of shape (batch, time).

        Returns:
            Tensor: The logits of shape (batch, time, 500).
        """

        # Embed then run the recurrence, and project every step
        embedded: Tensor = self.embedding(input_ids)
        outputs: Tensor
        hidden: Tensor
        outputs, hidden = self.gru(embedded)
        return self.fc(outputs)


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with a softmax attention pooling.
    """

    def __init__(self) -> None:
        """Init function."""

        # Init parent
        super().__init__()

        # The token embedding, the recurrence, the attention and the classifier
        self.embedding: nn.Embedding = nn.Embedding(1000, 32)
        self.lstm: nn.LSTM = nn.LSTM(
            32, 64, batch_first=True, bidirectional=True
        )
        self.attention: nn.Linear = nn.Linear(128, 1)
        self.fc: nn.Linear = nn.Linear(128, 10)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Attend and classify the token sequence.

        Args:
            input_ids (Tensor): The token ids of shape (batch, time).

        Returns:
            Tensor: The logits of shape (batch, 10).
        """

        # Embed then run the bidirectional recurrence
        embedded: Tensor = self.embedding(input_ids)
        outputs: Tensor
        hidden: tuple[Tensor, Tensor]
        outputs, hidden = self.lstm(embedded)

        # The attention weights over the time steps, then the pooled context
        scores: Tensor = self.attention(torch.tanh(outputs))
        weights: Tensor = F.softmax(scores, dim=1)
        context: Tensor = (outputs * weights).sum(dim=1)
        return self.fc(context)


class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder over grayscale images.
    """

    def __init__(self) -> None:
        """Init function."""

        # Init parent
        super().__init__()

        # The encoder and the decoder paths
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode then decode the images.

        Args:
            x (Tensor): The images of shape (batch, 1, 32, 32).

        Returns:
            Tensor: The reconstructions of shape (batch, 1, 32, 32).
        """

        # The latent code then the reconstruction
        latent: Tensor = self.encoder(x)
        return self.decoder(latent)


class Generator(nn.Module):
    """
    GAN generator from a latent vector to an image.
    """

    def __init__(self) -> None:
        """Init function."""

        # Init parent
        super().__init__()

        # The latent projection and the transposed convolutions
        self.fc: nn.Linear = nn.Linear(16, 128 * 4 * 4)
        self.net: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Generate images from the latent vectors.

        Args:
            z (Tensor): The latent vectors of shape (batch, 16).

        Returns:
            Tensor: The images of shape (batch, 3, 16, 16).
        """

        # Project, reshape and upsample
        projected: Tensor = self.fc(z)
        images: Tensor = projected.view(-1, 128, 4, 4)
        return self.net(images)


class Discriminator(nn.Module):
    """
    GAN discriminator from an image to a probability.
    """

    def __init__(self) -> None:
        """Init function."""

        # Init parent
        super().__init__()

        # The convolutions and the classifier head
        self.net: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.fc: nn.Linear = nn.Linear(64 * 4 * 4, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Score the realism of the images.

        Args:
            x (Tensor): The images of shape (batch, 3, 16, 16).

        Returns:
            Tensor: The probabilities of shape (batch, 1).
        """

        # Convolve then classify
        features: Tensor = self.net(x)
        return self.fc(features.view(features.size(0), -1))


class GCN(nn.Module):
    """
    Minimal graph convolutional network with dense adjacency.
    """

    def __init__(self) -> None:
        """Init function."""

        # Init parent
        super().__init__()

        # The per node transforms
        self.w1: nn.Parameter = nn.Parameter(torch.randn(8, 16) * 0.1)
        self.w2: nn.Parameter = nn.Parameter(torch.randn(16, 4) * 0.1)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Propagate the node features along the graph.

        Args:
            x (Tensor): The node features of shape (batch, nodes, 8).
            adj (Tensor): The adjacency matrices of shape (batch, nodes, nodes).

        Returns:
            Tensor: The node embeddings of shape (batch, nodes, 4).
        """

        # Two propagation rounds with a non linearity between them
        hidden: Tensor = torch.matmul(adj, torch.matmul(x, self.w1))
        hidden = F.relu(hidden)
        return torch.matmul(adj, torch.matmul(hidden, self.w2))


class ResNetTiny(nn.Module):
    """
    Tiny residual convolutional classifier.
    """

    def __init__(self) -> None:
        """Init function."""

        # Init parent
        super().__init__()

        # The stem and the residual block
        self.conv1: nn.Conv2d = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(16)
        self.conv2: nn.Conv2d = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(16)
        self.fc: nn.Linear = nn.Linear(16, 10)

    def forward(self, x: Tensor) -> Tensor:
        """
        Classify the images with a residual block.

        Args:
            x (Tensor): The images of shape (batch, 3, 32, 32).

        Returns:
            Tensor: The logits of shape (batch, 10).
        """

        # The stem, the residual connection and the classifier
        out: Tensor = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)) + out)
        pooled: Tensor = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(pooled)


def build_mlp() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build the MLP model with its dummy inputs."""

    # The model and its inputs
    model: MLPClassifier = MLPClassifier()
    inputs: dict[str, Tensor] = {"x": torch.randn(2, 784)}
    _train_one_step(model=model, inputs=inputs)
    return model, inputs


def build_lstm() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build the LSTM model with its dummy inputs."""

    # The model and its inputs
    model: LSTMClassifier = LSTMClassifier()
    inputs: dict[str, Tensor] = {"input_ids": torch.randint(0, 1000, (2, 12))}
    _train_one_step(model=model, inputs=inputs)
    return model, inputs


def build_rnn() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build the RNN model with its dummy inputs."""

    # The model and its inputs
    model: RNNClassifier = RNNClassifier()
    inputs: dict[str, Tensor] = {"input_ids": torch.randint(0, 1000, (2, 12))}
    _train_one_step(model=model, inputs=inputs)
    return model, inputs


def build_gru_seq2seq() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build the GRU sequence to sequence model with its dummy inputs."""

    # The model and its inputs
    model: GRUSeq2Seq = GRUSeq2Seq()
    inputs: dict[str, Tensor] = {"input_ids": torch.randint(0, 500, (2, 12))}
    _train_one_step(model=model, inputs=inputs)
    return model, inputs


def build_bilstm_attention() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build the bidirectional LSTM attention model with its dummy inputs."""

    # The model and its inputs
    model: BiLSTMAttention = BiLSTMAttention()
    inputs: dict[str, Tensor] = {"input_ids": torch.randint(0, 1000, (2, 12))}
    _train_one_step(model=model, inputs=inputs)
    return model, inputs


def build_conv_autoencoder() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build the convolutional autoencoder with its dummy inputs."""

    # The model and its inputs
    model: ConvAutoencoder = ConvAutoencoder()
    inputs: dict[str, Tensor] = {"x": torch.randn(2, 1, 32, 32)}
    _train_one_step(model=model, inputs=inputs)
    return model, inputs


def build_gan_generator() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build the GAN generator with its dummy inputs."""

    # The model and its inputs
    model: Generator = Generator()
    inputs: dict[str, Tensor] = {"z": torch.randn(2, 16)}
    _train_one_step(model=model, inputs=inputs)
    return model, inputs


def build_gan_discriminator() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build the GAN discriminator with its dummy inputs."""

    # The model and its inputs
    model: Discriminator = Discriminator()
    inputs: dict[str, Tensor] = {"x": torch.randn(2, 3, 16, 16)}
    _train_one_step(model=model, inputs=inputs)
    return model, inputs


def build_gcn() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build the graph convolutional network with its dummy inputs."""

    # The model and its inputs
    model: GCN = GCN()
    inputs: dict[str, Tensor] = {
        "x": torch.randn(2, 10, 8),
        "adj": torch.rand(2, 10, 10)
    }
    _train_one_step(model=model, inputs=inputs)
    return model, inputs


def build_resnet_cnn() -> tuple[nn.Module, dict[str, Tensor]]:
    """Build the tiny residual CNN with its dummy inputs."""

    # The model and its inputs
    model: ResNetTiny = ResNetTiny()
    inputs: dict[str, Tensor] = {"x": torch.randn(2, 3, 32, 32)}
    _train_one_step(model=model, inputs=inputs)
    return model, inputs


# The type of the local model builders
LocalBuilder = Callable[[], tuple[nn.Module, dict[str, Tensor]]]

# The local builders by model name
LOCAL_BUILDERS: dict[str, LocalBuilder] = {
    "local_mlp": build_mlp,
    "local_lstm": build_lstm,
    "local_rnn": build_rnn,
    "local_gru_seq2seq": build_gru_seq2seq,
    "local_bilstm_attention": build_bilstm_attention,
    "local_conv_autoencoder": build_conv_autoencoder,
    "local_gan_generator": build_gan_generator,
    "local_gan_discriminator": build_gan_discriminator,
    "local_gcn": build_gcn,
    "local_resnet_cnn": build_resnet_cnn
}
