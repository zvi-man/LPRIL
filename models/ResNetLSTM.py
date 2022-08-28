import os
from typing import Callable, Tuple
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, ResNet34_Weights, resnet34
import torch
from torchsummary import summary

# Constants
RESNET50_WEIGHTS_PATH = '.'
os.environ['TORCH_HOME'] = RESNET50_WEIGHTS_PATH
EXAMPLE_IMG_PATH = '/home/zvi/Desktop/dog.jpg'
RESNET_INPUT_SHAPE = (3, 224, 224)
NUM_OF_CHARS = 17


def freeze_model_learning(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


class ResnetLSTM(torch.nn.Module):
    def __init__(self, resnet_learn: bool = False, fc_before_size: int = 256,
                 lstm_hidden_size: int = 256, lstm_dropout: float = 0.0, bidirectional: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()
        full_resnet = resnet18(weights=weights)
        # full_resnet = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.resnet_without_last_layer = torch.nn.Sequential(*(list(full_resnet.children())[:-1]))
        if not resnet_learn:
            freeze_model_learning(self.resnet_without_last_layer)
        self.fc_before = torch.nn.Linear(in_features=512, out_features=fc_before_size)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = torch.nn.LSTM(input_size=fc_before_size, hidden_size=lstm_hidden_size, batch_first=True,
                                  dropout=lstm_dropout, bidirectional=bidirectional)
        self.d = 2 if bidirectional else 1
        self.fc_after = torch.nn.Linear(in_features=lstm_hidden_size * self.d, out_features=NUM_OF_CHARS)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.resnet_without_last_layer(x)
        x = torch.nn.Flatten()(x)
        x = self.fc_before(x)
        char_list = []
        hidden = (torch.zeros(self.d, batch_size, self.lstm_hidden_size),
                  torch.zeros(self.d, batch_size, self.lstm_hidden_size))
        for _ in range(3):
            out, hidden = self.lstm(x.unsqueeze(1), hidden)
            char = self.fc_after(out)
            char_list.append(char)
        output = torch.cat(char_list, dim=1)
        return output


def print_network_summary(model: torch.nn.Module, input_shape: Tuple) -> None:
    summary(model, input_shape)


if __name__ == '__main__':
    resnet = ResnetLSTM(resnet_learn=False, bidirectional=True)

    # Try random input
    rand_input = torch.rand(32, 3, 224, 224)
    rand_output = resnet(rand_input)
    print(f"input shape: {rand_input.shape}, output shape: {rand_output.shape}")
    # Try image
    img = read_image(EXAMPLE_IMG_PATH)
    batch = resnet.preprocess(img).unsqueeze(0)
    resnet(batch).squeeze(0)

    # Print network summary
    print_network_summary(resnet, RESNET_INPUT_SHAPE)

