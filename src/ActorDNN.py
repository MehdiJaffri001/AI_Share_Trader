import torch
import torch.nn as nn
from src.DeepNeuralNetwork import DeepNeuralNetwork

# extended Class created for actor networks that noutput data in a different format. Used by actor models only.

class ActorDNN(DeepNeuralNetwork):
    def __init__(self, inputSize, outputSize, dropoutRate, numberOfHiddenLayers, hiddenLayerSizes : list, numberOfStocks):
        super().__init__(inputSize, outputSize, dropoutRate, numberOfHiddenLayers, hiddenLayerSizes)
        
        self.logSTD = nn.Parameter(torch.ones(numberOfStocks) * -0.5)
    
    def forward(self, features):
        temp = super().forward(features)
        mean = torch.tanh(temp)
        stdDev = self.logSTD.exp()
        return mean, stdDev