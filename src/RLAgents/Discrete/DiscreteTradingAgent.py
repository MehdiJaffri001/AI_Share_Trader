from src.DeepNeuralNetwork import DeepNeuralNetwork
from src.RLAgents.RLTradingAgent import RLTradingAgent
from src.RLAgents.RLTradingAgent import EncoderCode
import torch
import torch.nn as nn
import numpy as np

# Parent RL class. manages executing actions for agents with discreet domain actions.
class DiscreteTradingAgent(RLTradingAgent):
    def __init__(self, env, initialCashBalance, encoderModel, encoderCode = EncoderCode.DNN.value):
        super().__init__(env, initialCashBalance, encoderModel, encoderCode = encoderCode)

    def executeDiscreteActionTrain(self, actions, dayNumber, currentBalance, numberOfShares: list):
        if dayNumber >= len(self.env.trainOpenPrices)-1:
            return None, 0, currentBalance, numberOfShares

        openPrices = self.env.trainOpenPrices[dayNumber]
        openPricesTMR = self.env.trainOpenPrices[dayNumber+1]
        return self.executeDiscreteAction(actions, dayNumber, currentBalance, numberOfShares, openPrices, openPricesTMR)
    
    def executeDiscreteActionVal(self, actions, dayNumber, currentBalance, numberOfShares: list):
        if dayNumber >= len(self.env.valOpenPrices)-1:
            return 0, currentBalance, numberOfShares

        openPrices = self.env.valOpenPrices[dayNumber]
        openPricesTMR = self.env.valOpenPrices[dayNumber+1]
        nextState, reward, currentBalance, numberOfShares = self.executeDiscreteAction(actions, dayNumber, currentBalance, numberOfShares, openPrices, openPricesTMR)

        return reward, currentBalance, numberOfShares

    def executeDiscreteActionTest(self, actions, dayNumber, currentBalance, numberOfShares: list):
        if dayNumber >= len(self.env.testOpenPrices)-1:
            return None, 0, currentBalance, numberOfShares

        openPrices = self.env.testOpenPrices[dayNumber]
        openPricesTMR = self.env.testOpenPrices[dayNumber+1]
        nextState, reward, currentBalance, numberOfShares = self.executeDiscreteAction(actions, dayNumber, currentBalance, numberOfShares, openPrices, openPricesTMR, True)
        return nextState, reward, currentBalance, numberOfShares
    
    def executeDiscreteAction(self, actions, dayNumber, currentBalance, numberOfShares: list, openPrices: list, nextDayOpenPrices: list, isTest = False):
        nanMask = ~(np.isnan(openPrices))
        sellMask = (actions == 2) & (numberOfShares > 0) & nanMask
        buyMask = (actions == 1) & nanMask

        # sell action
        sharesSold = np.zeros(self.numberOfStocks)
        sharesSold[sellMask] = np.minimum(100, numberOfShares[sellMask])
        self.tradeCount += sharesSold[sellMask].sum()
        balanceGained = (1-self.env.transactionFee) * sharesSold[sellMask] * openPrices[sellMask]
        currentBalance += balanceGained.sum()
        numberOfShares[sellMask] -= sharesSold[sellMask]

        # buy action
        stockPrice = np.zeros(self.numberOfStocks)
        stockPrice[buyMask] = openPrices[buyMask] * (1+self.env.transactionFee)
        totalCost = stockPrice.sum() + 1e-8

        percentageShare = np.zeros(self.numberOfStocks)
        percentageShare[buyMask] = stockPrice[buyMask] / totalCost
        
        sharesBought = np.zeros(self.numberOfStocks)
        sharesBought[buyMask] = (percentageShare[buyMask] * currentBalance) / stockPrice[buyMask]
        sharesBought = sharesBought.astype(int) 
        
        sharesBought[buyMask] = np.minimum(sharesBought[buyMask], 100)
        sharesBought[buyMask] = np.minimum(sharesBought[buyMask], 1000 - numberOfShares[buyMask])
        currentBalance -= (sharesBought[buyMask] * stockPrice[buyMask]).sum()
        sharesBought = sharesBought.astype(int)
        self.tradeCount += sharesBought[buyMask].sum()
        numberOfShares[buyMask] += sharesBought[buyMask]

        # calc new portfolio value
        convertedNextDayOpenPrices = np.nan_to_num(nextDayOpenPrices, nan=0.0)
        totalPortfolioValue = currentBalance + np.sum(numberOfShares * convertedNextDayOpenPrices)
        reward = (totalPortfolioValue/self.initialCashBalance)
        if isTest:
            nextState = self.getStateTest(dayNumber+1, currentBalance, numberOfShares)
        else:
            nextState = self.getState(dayNumber+1, currentBalance, numberOfShares)

        return nextState, reward, currentBalance, numberOfShares