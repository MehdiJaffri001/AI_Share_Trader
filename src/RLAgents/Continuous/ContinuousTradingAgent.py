from src.DeepNeuralNetwork import DeepNeuralNetwork
from src.RLAgents.RLTradingAgent import RLTradingAgent
from src.RLAgents.RLTradingAgent import EncoderCode
import torch
import torch.nn as nn
import numpy as np

# Parent RL class. manages executing actions for agents with continuous domain actions.
class ContinuousTradingAgent(RLTradingAgent):
    def __init__(self, env, initialCashBalance, encoderModel, encoderCode = EncoderCode.DNN.value):
        super().__init__(env, initialCashBalance, encoderModel, encoderCode = encoderCode)

    def executeContinuousActionTrain(self, actions, dayNumber, currentBalance, numberOfShares: list):
        if dayNumber >= len(self.env.trainOpenPrices)-1:
            return 0, currentBalance, numberOfShares

        openPrices = self.env.trainOpenPrices[dayNumber]
        openPricesTMR = self.env.trainOpenPrices[dayNumber+1]
        reward, currentBalance, numberOfShares = self.executeContinuousAction(actions, dayNumber, currentBalance, numberOfShares, openPrices, openPricesTMR)
        return reward, currentBalance, numberOfShares

    def executeContinuousActionVal(self, actions, dayNumber, currentBalance, numberOfShares: list):
        if dayNumber >= len(self.env.valOpenPrices)-1:
            return 0, currentBalance, numberOfShares

        openPrices = self.env.valOpenPrices[dayNumber]
        openPricesTMR = self.env.valOpenPrices[dayNumber+1]
        reward, currentBalance, numberOfShares = self.executeContinuousAction(actions, dayNumber, currentBalance, numberOfShares, openPrices, openPricesTMR)
        return reward, currentBalance, numberOfShares

    def executeContinuousActionTest(self, actions, dayNumber, currentBalance, numberOfShares: list):
        if dayNumber >= len(self.env.testOpenPrices)-1:
            return 0, currentBalance, numberOfShares

        openPrices = self.env.testOpenPrices[dayNumber]
        openPricesTMR = self.env.testOpenPrices[dayNumber+1]
        reward, currentBalance, numberOfShares = self.executeContinuousAction(actions, dayNumber, currentBalance, numberOfShares, openPrices, openPricesTMR)
        return reward, currentBalance, numberOfShares
        
    def executeContinuousAction(self, actions, dayNumber, currentBalance, numberOfShares: list, openPrices: list, nextDayOpenPrices: list):        
        nanMask = ~(np.isnan(openPrices))
        sellMask = (actions < 0) & (numberOfShares > 0) & nanMask
        buyMask = (actions > 0) & nanMask

        # sell action
        sharesSold = np.zeros(self.numberOfStocks)
        sharesSold[sellMask] = np.minimum(100 * -actions[sellMask], numberOfShares[sellMask])
        sharesSold = sharesSold.astype(int)
        self.tradeCount += sharesSold[sellMask].sum()
        #print("selling", sharesSold[sellMask].sum(), " stocks, share count is: ", sharesSold[sellMask])
        self.tradeCountSell += sharesSold[sellMask].sum()
        balanceGained = (1-self.env.transactionFee) * sharesSold[sellMask] * openPrices[sellMask]
        currentBalance += balanceGained.sum()
        numberOfShares[sellMask] -= sharesSold[sellMask]
        #self.sharesTradedSell = []

        #self.sharesTradedBuy = []

        
        
        # buy action
        stockPrice = np.zeros(self.numberOfStocks)
        stockPrice[buyMask] =  actions[buyMask] * 100 * openPrices[buyMask] * (1+self.env.transactionFee)
        totalCost = stockPrice.sum() + 1e-8

        percentageCost = currentBalance/totalCost # reveals whether we can buy all the stocks we want or not.
        percentageCost = min(percentageCost, 1) # if we have enough money, percentage cost is 1, and we buy all stocks, otherwise percCost is some fraction like 0.5 and then we can only buy 0.5 of the stocks we want.
        
        percentageShare = np.zeros(self.numberOfStocks)
        percentageShare[buyMask] = stockPrice[buyMask] * percentageCost
        
        sharesBought = np.zeros(self.numberOfStocks)
        sharesBought[buyMask] = (percentageShare[buyMask]) / (openPrices[buyMask] * (1+self.env.transactionFee))
        sharesBought[buyMask] = np.minimum(sharesBought[buyMask], 100)
        sharesBought[buyMask] = np.minimum(sharesBought[buyMask], 1000 - numberOfShares[buyMask])
        sharesBought = sharesBought.astype(int)
        currentBalance -= (sharesBought[buyMask] * openPrices[buyMask] * (1+self.env.transactionFee)).sum()
        numberOfShares[buyMask] += sharesBought[buyMask]
        self.tradeCount += sharesBought[buyMask].sum()
        self.tradeCountBuy += sharesBought[buyMask].sum()
        #print("Buying", sharesBought[buyMask].sum(), " stocks, share count is: ", sharesBought[buyMask])
        
        # calc new portfolio value
        convertedNextDayOpenPrices = np.nan_to_num(nextDayOpenPrices, nan=0.0)
        totalPortfolioValue = currentBalance + np.sum(numberOfShares * convertedNextDayOpenPrices)
        reward = (totalPortfolioValue/self.initialCashBalance)

        #print("Rem bal", currentBalance)
        
        return reward, currentBalance, numberOfShares