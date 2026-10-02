import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import random

import math

from src.Environment.StockEnvironment import StockEnvironment
from src.DeepNeuralNetwork import DeepNeuralNetwork

# generates the DNN encoder
class EncoderGenerator:
    def __init__(self, numberOfEpochs = 200, env = StockEnvironment(), savePath = "data/dataModels"):
        print("Initialising Stock Trading DNN encoder.")
        self.seeds = [41, 101, 777, 999, 2026]
        self.epochs = numberOfEpochs
        self.baselineStatsFilename = savePath + 'baselineModelStats.txt'
        self.env = env
        self.stableModelName = savePath + 'DNNEncoder_stable.pth'
        self.greedyModelName = savePath + 'DNNEncoder_greedy.pth'
        self.balancedModelName = savePath + 'DNNEncoder_balanced.pth'
    
    def generateModels(self):
        self.greedyModelROI = 1.0
        self.greedyModelLoss = 1.0

        self.stableModelLoss = 1.0
        self.stableModelROI = 1.0

        self.balancedModelLoss = 1.0
        self.balancedModelROI = 1.0

        print("Starting generation of seeded models.")
        
        for seed in self.seeds:
            random.seed(seed)
            np.random.default_rng(seed)
            torch.manual_seed(seed)
            self.env.seedEnv(seed)
            self.env.createLoader(shuffleTrain = True, batchSize = 32, formatStaticGraph = False)
            print(f"Training Model using seed: {seed} ")
            self.train(numEpochs = self.epochs)
        
        with open(self.baselineStatsFilename, 'w') as file:
            file.write(f"Stable Model achieved ROI: {self.stableModelROI:.3f} with loss {self.stableModelLoss:.6f}\n")
            file.write(f"Greedy ROI achieved: {self.greedyModelROI:.3f} with loss {self.greedyModelLoss:.6f}\n")
            file.write(f"Balanced Model achieved ROI: {self.balancedModelROI:.3f} with loss {self.balancedModelLoss:.6f}")

        print(f"Stable Model achieved ROI: {self.stableModelROI:.3f} with loss {self.stableModelLoss:.6f}")
        print(f"Best ROI achieved: {self.greedyModelROI:.3f} with loss {self.greedyModelLoss:.6f}")
        print(f"Oportunistic Model achieved ROI: {self.balancedModelROI:.3f} with loss {self.balancedModelLoss:.6f}")

    def train(self, numEpochs = 200, learningRate = 0.0003, weightDecay = 1e-5, initialCash = 1000000, buyThreshold = 0.0035):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Train the model
        model = DeepNeuralNetwork(self.env.flattenedNumberOfFeatures,self. env.numberOfStocks, 0.10, 3, [256, 128, 64])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        for epoch in range(numEpochs):
            model.train()
            totalTrainLoss = 0
    
            for index, (features, targets, mask) in enumerate(self.env.trainLoader):

                features = features.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
        
                optimizer.zero_grad()
        
                # Forward pass
                outputs = model(features)
                       
                squaredErrors = ((outputs * mask) - (targets * mask)) ** 2
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
                    for valFeatures, valTargets, valMask in self.env.valLoader:
                        valFeatures = valFeatures.to(device)
                        valTargets = valTargets.to(device)
                        valMask = valMask.to(device)
                
                        valOutputs = model(valFeatures)

                        valMaskedOutput = valOutputs * valMask
                        valMaskedTarget = valTargets * valMask

                        batchPredictions  = valMaskedOutput.cpu().numpy()
                        batchTargets  = valMaskedTarget.cpu().numpy()
                        batchMasks  = valMask.cpu().numpy()
                
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
                    torch.save(model.state_dict(), self.stableModelName)
                elif avgValLoss == self.stableModelLoss:
                    if self.stableModelROI < ROI:
                        self.stableModelROI = ROI
                        torch.save(model.state_dict(), self.stableModelName)

                if self.greedyModelROI < ROI:
                    self.greedyModelROI = ROI
                    self.greedyModelLoss = float(avgValLoss)
                    torch.save(model.state_dict(), self.greedyModelName)

                if ROI/avgValLoss > self.balancedModelROI/self.balancedModelLoss:
                    self.balancedModelLoss = avgValLoss
                    self.balancedModelROI = ROI
                    torch.save(model.state_dict(), self.balancedModelName)
        
                print(f"--- Epoch {epoch+1} Summary | ROI: {ROI:.3f} | Avg Train Loss: {avgTrainLoss:.6f} | Avg Val Loss: {avgValLoss:.6f} ---")

                scheduler.step(avgValLoss)

if __name__ == "__main__":
    encoderGen = EncoderGenerator()
    encoderGen.generateModels()