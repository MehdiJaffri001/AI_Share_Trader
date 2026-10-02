from src.DeepNeuralNetwork import DeepNeuralNetwork
import torch
import torch.nn as nn
import numpy as np
import copy

from tsl.ops.connectivity import adj_to_edge_index

from enum import Enum

# enum class for encoder management
class EncoderCode(Enum):
    DNN = 1
    GWN = 2
    AGCRN = 3

# GrandParent RL class. manages state and processing data using the encoder so that it is ready for used by the children RL agents.
class RLTradingAgent:
    def __init__(self, env, initialCashBalance, encoderModel, encoderCode = EncoderCode.DNN.value):
        self.initialCashBalance = initialCashBalance
        self.env = env
        self.numberOfStocks = self.env.numberOfStocks
        self.encoder = encoderModel
        self.encoderWeights = encoderModel.state_dict()
        self.tradeCount = 0
        self.tradeCountBuy = 0
        self.tradeCountSell = 0
        self.sharesTradedBuy = []
        self.sharesTradedSell = []
        if encoderCode == EncoderCode.DNN.value:
            self.calcEncodedDataDNN()
        else:
            self.calcEncodedDataSTGNN(encoderCode)

    def resetEncoderWeights():
            self.encoder.load_state_dict(self.encoderWeights)
    
    def getState(self, dayNumber, currentBalance, numberOfSharesOwned : list):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sharesOwned = torch.as_tensor(np.array(numberOfSharesOwned) / 1000.0, dtype=torch.float32, device=device)
        balance = torch.as_tensor([currentBalance / self.initialCashBalance], dtype=torch.float32, device=device)

        # DDQN input is stock predictions from encoder, then the current number of stocks and then the current balance.
        encoderDataTensor = self.encodedDataTrain[dayNumber]
        state = torch.cat([encoderDataTensor, sharesOwned, balance])
        return state

    def getStateVal(self, dayNumber, currentBalance, numberOfSharesOwned : list):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sharesOwned = torch.tensor(np.array(numberOfSharesOwned) / 1000.0, dtype=torch.float32, device=device)
        balance = torch.tensor([currentBalance / self.initialCashBalance], dtype=torch.float32, device=device)

        # DDQN input is stock predictions from encoder, then the current number of stocks and then the current balance.
        encoderDataTensor = self.encodedDataValidation[dayNumber]
        state = torch.cat([encoderDataTensor, sharesOwned, balance])
        return state

    def getStateTest(self, dayNumber, currentBalance, numberOfSharesOwned : list):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sharesOwned = torch.tensor(np.array(numberOfSharesOwned) / 1000.0, dtype=torch.float32, device=device)
        balance = torch.tensor([currentBalance / self.initialCashBalance], dtype=torch.float32, device=device)

        # DDQN input is stock predictions from encoder, then the current number of stocks and then the current balance.
        encoderDataTensor = self.encodedDataTest[dayNumber]
        state = torch.cat([encoderDataTensor, sharesOwned, balance])
        return state
    
    def calcEncodedDataDNN(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encodedDataTrain = []
        self.encodedDataValidation = []
        self.encodedDataTest = []
        self.encoder = self.encoder.to(device)
        
        with torch.no_grad():
            # train data
            for dFeatures, dTargets, dMasks in self.env.trainLoader:
                dFeatures = dFeatures.to(device)
                dMasks = dMasks.to(device)
                
                encoderOutputs = self.encoder(dFeatures) # get encoded data for today
                encoderOutputs = encoderOutputs * dMasks # remove unavailable stocks

                batchedEncodedData = encoderOutputs.cpu().numpy()

                for batchIndex in range(batchedEncodedData.shape[0]):
                    self.encodedDataTrain.append(batchedEncodedData[batchIndex])

            # val data
            for dFeatures, dTargets, dMasks in self.env.valLoader:
                dFeatures = dFeatures.to(device)
                dMasks = dMasks.to(device)
                
                encoderOutputs = self.encoder(dFeatures) # get encoded data for today
                encoderOutputs = encoderOutputs * dMasks # remove unavailable stocks

                batchedEncodedData = encoderOutputs.cpu().numpy()

                for batchIndex in range(batchedEncodedData.shape[0]):
                    self.encodedDataValidation.append(batchedEncodedData[batchIndex])

            # test data
            for dFeatures, dTargets, dMasks in self.env.testLoader:
                dFeatures = dFeatures.to(device)
                dMasks = dMasks.to(device)
                
                encoderOutputs = self.encoder(dFeatures) # get encoded data for today
                encoderOutputs = encoderOutputs * dMasks # remove unavailable stocks

                batchedEncodedData = encoderOutputs.cpu().numpy()

                for batchIndex in range(batchedEncodedData.shape[0]):
                    self.encodedDataTest.append(batchedEncodedData[batchIndex])
            
        self.encodedDataTrain = np.array(self.encodedDataTrain)
        self.encodedDataValidation = np.array(self.encodedDataValidation)
        self.encodedDataTest = np.array(self.encodedDataTest)

        self.encodedDataTrain = torch.tensor(self.encodedDataTrain, dtype=torch.float32, device=device)
        self.encodedDataValidation = torch.tensor(self.encodedDataValidation, dtype=torch.float32, device=device)
        self.encodedDataTest = torch.tensor(self.encodedDataTest, dtype=torch.float32, device=device)
        
    def calcEncodedDataSTGNN(self, encoderCode = EncoderCode.GWN.value):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encodedDataTrain = []
        self.encodedDataValidation = []
        self.encodedDataTest = []
        self.encoder = self.encoder.to(device)

        adjacencyMatrix = np.ones((self.numberOfStocks, self.numberOfStocks))
        edgeIndices, edgeWeights = adj_to_edge_index(adjacencyMatrix)

        edgeIndices = torch.tensor(edgeIndices, dtype=torch.long)
        edgeWeights = torch.tensor(edgeWeights, dtype=torch.float)

        edgeIndices = edgeIndices.to(device)
        edgeWeights = edgeWeights.to(device)

        with torch.no_grad():
            # train data
            for batch in self.env.trainLoader:
                
                x = batch.x.to(device)
                mask = batch.mask.to(device)
        
                # Forward pass
                if EncoderCode.GWN.value == encoderCode:
                    outputs = self.encoder(x=x, edge_index=edgeIndices, edge_weight=edgeWeights)
                elif EncoderCode.AGCRN.value == encoderCode:
                    outputs = self.encoder(x=x)

                batchedEncodedData = outputs.cpu().numpy()

                for batchIndex in range(batchedEncodedData.shape[0]):
                    self.encodedDataTrain.append(batchedEncodedData[batchIndex])

            # val data
            for batch in self.env.valLoader:

                x = batch.x.to(device)
                mask = batch.mask.to(device)
                
                # Forward pass
                if EncoderCode.GWN.value == encoderCode:
                    outputs = self.encoder(x=x, edge_index=edgeIndices, edge_weight=edgeWeights)
                elif EncoderCode.AGCRN.value == encoderCode:
                    outputs = self.encoder(x=x)

                batchedEncodedData = outputs.cpu().numpy()

                for batchIndex in range(batchedEncodedData.shape[0]):
                    self.encodedDataValidation.append(batchedEncodedData[batchIndex])

            # test data
            for batch in self.env.testLoader:

                x = batch.x.to(device)
                mask = batch.mask.to(device)
                
                # Forward pass
                if EncoderCode.GWN.value == encoderCode:
                    outputs = self.encoder(x=x, edge_index=edgeIndices, edge_weight=edgeWeights)
                elif EncoderCode.AGCRN.value == encoderCode:
                    outputs = self.encoder(x=x)

                batchedEncodedData = outputs.cpu().numpy()

                for batchIndex in range(batchedEncodedData.shape[0]):
                    self.encodedDataTest.append(batchedEncodedData[batchIndex])
            
        self.encodedDataTrain = np.array(self.encodedDataTrain).squeeze()
        self.encodedDataValidation = np.array(self.encodedDataValidation).squeeze()
        self.encodedDataTest = np.array(self.encodedDataTest).squeeze()

        #print("Done encoding data. output shapes:")
        #print(f"Train shape : {self.encodedDataTrain.shape}")
        #print(f"Val shape : {self.encodedDataValidation.shape}")
        #print(f"Test shape : {self.encodedDataTest.shape}")
        
        self.encodedDataTrain = torch.tensor(self.encodedDataTrain, dtype=torch.float32, device=device)
        self.encodedDataValidation = torch.tensor(self.encodedDataValidation, dtype=torch.float32, device=device)
        self.encodedDataTest = torch.tensor(self.encodedDataTest, dtype=torch.float32, device=device)
        