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


class Resnet50LSTM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        full_resnet = resnet18(weights=weights)
        self.resnet_without_last_layer = torch.nn.Sequential(*(list(full_resnet.children())[:-1]))
        # self.resnet50 = resnet50()
        self.fc = torch.nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        x = self.resnet_without_last_layer(x)
        x = torch.nn.Flatten()(x)
        print(x.shape)
        return self.fc(x)


def try_image_on_model(model) -> None:
    preprocess = model.weights.transforms()

    img = read_image(EXAMPLE_IMG_PATH)
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)


def print_network_summary(model: Callable, input_shape: Tuple) -> None:
    summary(model, input_shape)


if __name__ == '__main__':
    re50 = Resnet50LSTM()
    print_network_summary(re50, RESNET_INPUT_SHAPE)
