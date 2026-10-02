import torch
import torch.nn as nn

# Class created to simplify dnn creation. Used by all rl models.

class DeepNeuralNetwork(nn.Module):
    def __init__(self, inputSize, outputSize, dropoutRate, numberOfHiddenLayers, hiddenLayerSizes : list):
        super(DeepNeuralNetwork, self).__init__()

        tempLayers = []
        
        if 0 >= numberOfHiddenLayers:
            tempLayers.append(nn.Linear(inputSize, outputSize))
        else:
            tempLayers.append(nn.Linear(inputSize, hiddenLayerSizes[0]))
            tempLayers.append(nn.ReLU())
            tempLayers.append(nn.Dropout(dropoutRate))

            for index in range(numberOfHiddenLayers-1):
                tempLayers.append(nn.Linear(hiddenLayerSizes[index], hiddenLayerSizes[index+1]))
                tempLayers.append(nn.ReLU())
                tempLayers.append(nn.Dropout(dropoutRate))

            tempLayers.append(nn.Linear(hiddenLayerSizes[numberOfHiddenLayers-1], outputSize))

        self.layers = nn.ModuleList(tempLayers)
    
    def forward(self, features):
        temp = features
        for layer in self.layers:
            temp = layer(temp) 
        return temp