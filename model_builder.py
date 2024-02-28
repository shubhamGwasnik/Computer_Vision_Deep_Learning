
"""Contains Pytorch model code to initiatiate a TinyVGG model from cnn explainer"""

import torch.nn as nn

class TinyVGG(nn.Module):
    """Model architecture : copying tiny VGG from CNN explainer
    Args: 
        input_shape: an int indicating no of Inpout channels
        hidden_units: an int indicating no of hidden units between layers
        output_shape: An integer indicating no of output units
    Returns: 
        model
    """
    def __init__(self,input_shape:int, hidden_units:int, output_shape:int) -> None:
        super().__init__()
        self.conv_block_1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=0,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # default stride ==kernel_size
        )
        self.conv_block_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # default stride ==kernel_size

        )
        self.classifier=nn.Sequential(nn.Flatten(),
                                      nn.Linear(in_features=hidden_units*13*13, out_features=output_shape))
    def forward(self,x):
        # # x=self.conv_block_1(x)
        # print(x.shape)
        # x=self.conv_block_2(x)
        # print(x.shape)
        # x=self.classifier(x)
        # print(x.shape)
        # return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))  # benefits from operator fusion

