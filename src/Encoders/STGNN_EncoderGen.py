import torch
import torch.nn as nn

import numpy as np
import random

import math

from tsl.nn.models import GraphWaveNetModel
from tsl.nn.models import AGCRNModel
from tsl.ops.connectivity import adj_to_edge_index

from src.Environment.StockEnvironment import StockEnvironment
from src.DeepNeuralNetwork import DeepNeuralNetwork
from src.RLAgents.RLTradingAgent import EncoderCode

# generates the GWN or AGCRN encoder
class STGNNEncoderGenerator:
    def __init__(self, numberOfEpochs = 200, env = StockEnvironment(useJSE = True, flattenedNodes = False), savePath = "data/modelData/", encoderCode = EncoderCode.GWN.value):
        print("Initialising Stock Trading STGNN encoder.")
        self.seeds = [41, 101, 777, 999, 2026]
        self.epochs = numberOfEpochs
        self.env = env
        self.encoderCode = encoderCode
        print(f"Generating {EncoderCode(encoderCode).name} encoder")
        if EncoderCode.GWN.value == encoderCode:
            self.baselineStatsFilename = savePath + 'GWNbaselineModelStats.txt'

            self.stableModelSavePath = savePath + 'GWNEncoder_stable.pth'
            self.greedyModelSavePath = savePath + 'GWNEncoder_greedy.pth'
            self.balancedModelSavePath = savePath + 'GWNEncoder_balanced.pth'
        elif EncoderCode.AGCRN.value == encoderCode:
            self.baselineStatsFilename = savePath + 'AGCRNbaselineModelStats.txt'

            self.stableModelSavePath = savePath + 'AGCRNEncoder_stable.pth'
            self.greedyModelSavePath = savePath + 'AGCRNEncoder_greedy.pth'
            self.balancedModelSavePath = savePath + 'AGCRNEncoder_balanced.pth'
    
    def generateModels(self):

        self.stableModelLoss = 1.0
        self.stableModelROI = 1.0
        self.balancedModelLoss = 1.0
        self.balancedModelROI = 1.0
        self.greedyModelLoss = 1.0
        self.greedyModelROI = 1.0

        print("Starting generation of seeded STGNN Encoder models.")
        
        for seed in self.seeds:
            random.seed(seed)
            np.random.default_rng(seed)
            torch.manual_seed(seed)
            self.env.seedEnv(seed)
            self.env.createLoader(shuffleTrain = False, batchSize = 32, formatStaticGraph = True)
            print(f"Training Model using seed: {seed} ")
            self.train(numEpochs = self.epochs)
        
        with open(self.baselineStatsFilename, 'w') as file:
            file.write(f"Stable Model achieved ROI: {self.stableModelROI:.3f} with loss {self.stableModelLoss:.6f}\n")
            file.write(f"Greedy ROI achieved: {self.greedyModelROI:.3f} with loss {self.greedyModelLoss:.6f}\n")
            file.write(f"Balanced ROI achieved: {self.balancedModelROI:.3f} with loss {self.balancedModelLoss:.6f}\n")

        print(f"Stable Model achieved ROI: {self.stableModelROI:.3f} with loss {self.stableModelLoss:.6f}\n")
        print(f"Greedy ROI achieved: {self.greedyModelROI:.3f} with loss {self.greedyModelLoss:.6f}\n")
        print(f"Balanced ROI achieved: {self.balancedModelROI:.3f} with loss {self.balancedModelLoss:.6f}\n")

    def train(self, numEpochs = 200, learningRate = 0.0003, weightDecay = 1e-5, initialCash = 1000000, buyThreshold = 0.0035):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        adjacencyMatrix = np.ones((self.env.numberOfStocks, self.env.numberOfStocks))
        edgeIndices, edgeWeights = adj_to_edge_index(adjacencyMatrix)

        edgeIndices = torch.tensor(edgeIndices, dtype=torch.long)
        edgeWeights = torch.tensor(edgeWeights, dtype=torch.float)

        edgeIndices = edgeIndices.to(device)
        edgeWeights = edgeWeights.to(device)

        minLoss = 1
        
        if EncoderCode.GWN.value == self.encoderCode:
            model = GraphWaveNetModel(
                input_size = 8, # hardcoded number of features + 1
                output_size = 1,
                horizon = 1,
                n_nodes = self.env.numberOfStocks,
                learned_adjacency = True,
                emb_size = 12,
                hidden_size = 32
            )
        elif EncoderCode.AGCRN.value == self.encoderCode:
            model = AGCRNModel(
                input_size = 8,
                output_size = 1,
                horizon = 1,
                n_nodes = self.env.numberOfStocks,
                emb_size = 12,
                hidden_size = 32,
                n_layers = 1
            )
        model = model.to(device)
    
        # Train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)

        valLossLessCount = 0
        valLossLessCountStop = 5
        
        for epoch in range(numEpochs):
            model.train()
            totalTrainLoss = 0

            print(f"Training - Current Epoch: {epoch}\r", end = '')
    
            for batch in self.env.trainLoader:

                x = batch.x.to(device)
                y = batch.y.to(device)
                mask = batch.mask.to(device)
        
                optimizer.zero_grad()
        
                # Forward pass
                if EncoderCode.GWN.value == self.encoderCode:
                    outputs = model(x=x, edge_index=edgeIndices, edge_weight=edgeWeights)
                elif EncoderCode.AGCRN.value == self.encoderCode:
                    outputs = model(x=x)

                squaredErrors = ((outputs * mask) - (y * mask)) ** 2
                loss = squaredErrors.sum() / (mask.sum() + 1e-8)
            
                loss.backward()
                optimizer.step()

                totalTrainLoss += loss.item()

            if (0 == ((epoch+1) % 5)):
                model.eval()
                totalValLoss = 0
    
                with torch.no_grad():
                    totalCash = initialCash
                    dayNumber = 0
                    numberOfSharesOwned = np.zeros(self.env.numberOfStocks)
                    lastPrice = np.copy(self.env.valOpenPrices[0])
                    for valBatch in self.env.valLoader:

                        valX = valBatch.x.to(device)
                        valY = valBatch.y.to(device)
                        valMask = valBatch.mask.to(device)  

                        if EncoderCode.GWN.value == self.encoderCode:
                            valOutputs = model(x=valX, edge_index=edgeIndices, edge_weight=edgeWeights)
                        elif EncoderCode.AGCRN.value == self.encoderCode:
                            valOutputs = model(x=valX)

                        valMaskedOutput = valOutputs * valMask
                        valMaskedTarget = valY * valMask

                        batchPredictions = valOutputs.squeeze(1).squeeze(-1).cpu().numpy()
                        batchTargets = valY.squeeze(1).squeeze(-1).cpu().numpy()
                        batchMasks = valMask.squeeze(1).squeeze(-1).cpu().numpy()
                
                        for dayPredictions, dayTargets, dayMasks in zip(batchPredictions, batchTargets, batchMasks):

                            if dayNumber >= len(self.env.valOpenPrices):
                                break
                            sortedStockIndexes = np.argsort(dayPredictions)[::-1]
                    
                            for stockIndex in sortedStockIndexes:
                                actualReturn = dayTargets[stockIndex]
                                openPrice = self.env.valOpenPrices[dayNumber][stockIndex]
                                if openPrice <= 0:
                                    continue

                                if (not np.isnan(dayTargets[stockIndex])) and (not np.isnan(self.env.valOpenPrices[dayNumber][stockIndex])):
                                    lastPrice[stockIndex] = self.env.valOpenPrices[dayNumber][stockIndex] * (1+dayTargets[stockIndex]) 

                                if 0 == dayMasks[stockIndex] or np.isnan(openPrice):
                                    continue
                                elif dayPredictions[stockIndex] > buyThreshold:
                                    numberOfSharesBought = min(math.floor(totalCash//openPrice), 100)

                                    if 0 < numberOfSharesBought:
                                        cost = numberOfSharesBought*openPrice
                                        totalCash -= cost * (1+self.env.transactionFee)
                                        numberOfSharesOwned[stockIndex] += numberOfSharesBought
                                elif dayPredictions[stockIndex] < -buyThreshold and numberOfSharesOwned[stockIndex] > 0:
                                    numberOfSharesSold = min(numberOfSharesOwned[stockIndex], 100)

                                    if 0 < numberOfSharesSold:
                                        cost = numberOfSharesSold*openPrice
                                        totalCash += cost * (1-self.env.transactionFee)
                                        numberOfSharesOwned[stockIndex] -= numberOfSharesSold
                                
                            dayNumber+=1
                
                        valSquaredErrors = (valMaskedOutput - valMaskedTarget) ** 2
                        valLoss = valSquaredErrors.sum() / (valMask.sum() + 1e-8)
                
                        totalValLoss += valLoss.item()
                    # trade period is over
                    for stockIndex in range(self.env.numberOfStocks):
                        if (numberOfSharesOwned[stockIndex] > 0):
                            finalStockValue = lastPrice[stockIndex]
                            if (not np.isnan(finalStockValue)) and finalStockValue > 0:
                                totalCash += numberOfSharesOwned[stockIndex] * finalStockValue
                
                ROI = totalCash/initialCash
                # Calculate average losses across the datasets
                avgTrainLoss = totalTrainLoss / len(self.env.trainLoader)
                avgValLoss = totalValLoss / len(self.env.valLoader)
                
                if avgValLoss < self.stableModelLoss:
                    self.stableModelLoss = float(avgValLoss)
                    self.stableModelROI = ROI
                    torch.save(model.state_dict(), self.stableModelSavePath)
                elif avgValLoss == self.stableModelLoss:
                    if self.stableModelROI < ROI:
                        self.stableModelROI = ROI
                        torch.save(model.state_dict(), self.stableModelSavePath)

                if avgValLoss < minLoss:
                    minLoss = avgValLoss
                    valLossLessCount = 0
                else:
                    valLossLessCount += 1

                if self.greedyModelROI < ROI:
                    self.greedyModelROI = ROI
                    self.greedyModelLoss = float(avgValLoss)
                    torch.save(model.state_dict(), self.greedyModelSavePath)

                if avgValLoss/ROI < self.balancedModelLoss/self.balancedModelROI:
                    self.balancedModelLoss = avgValLoss
                    self.balancedModelROI = ROI
                    torch.save(model.state_dict(), self.balancedModelSavePath)    
                
                print(f"--- Epoch {epoch+1} Summary | ROI: {ROI:.3f} | Avg Train Loss: {avgTrainLoss:.6f} | Avg Val Loss: {avgValLoss:.6f} ---")

                if valLossLessCount > valLossLessCountStop:
                    print(f"Stopping training early, since avg Val loss hasn't improved for {valLossLessCountStop*5} epochs.")
                    break
    
if __name__ == "__main__":
    encoderGen = STGNNEncoderGenerator()
    encoderGen.generateModels()