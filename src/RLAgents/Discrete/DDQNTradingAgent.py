import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.RLAgents.Discrete.DiscreteTradingAgent import DiscreteTradingAgent
from src.RLAgents.RLTradingAgent import EncoderCode
from collections import namedtuple
from src.DeepNeuralNetwork import DeepNeuralNetwork

import numpy as np
import random
import gc
import math

# DDQN class
class DDQNTradingAgent(DiscreteTradingAgent):
    def __init__(self, env, encoderModel, encoderCode = EncoderCode.DNN.value, initialCashBalance = 1000000, epochs = 500):
        super().__init__(env, initialCashBalance, encoderModel, encoderCode = encoderCode)
        self.epochs = epochs

    # trains model over the train set and saves based on val set performance
    def train(self, modelROI_Threshold = 0, modelSavePathGreedy = 'DDQN_greedy.pth', modelSavePathBalanced = 'DDQN_balanced.pth', env= None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if (not None == env):
            self.env = env
        
        learningRate = 0.0005
        weightDecay = 1e-5
        minNumberForReplay = 1000
        maxNumberForReplay = 50000
        bellmanCoefficient = 0.99

        highestRatioAchieved = 0
        highestROIAchieved = 0
        
        epsStart = 0.9
        epsEnd = 0.0001
        epsDecay = 0.99
        currentEps = epsStart
        
        # for the JSE: 34 stocks :       input : 69,                output: 102, dropout: 0.2, hidden layers 128, 128 and 128  
        modelPolicy = DeepNeuralNetwork(self.numberOfStocks*2 + 1, self.numberOfStocks*3, 0.2, 3, [128, 128, 128])
        modelPolicy = modelPolicy.to(device)

        modelTarget = DeepNeuralNetwork(self.numberOfStocks*2 + 1, self.numberOfStocks*3, 0.2, 3, [128, 128, 128])
        modelTarget = modelTarget.to(device)
        modelTarget.load_state_dict(modelPolicy.state_dict())

        modelTarget.eval()
        
        optimizer = torch.optim.Adam(modelPolicy.parameters(), lr=learningRate, weight_decay=weightDecay)

        lossCriterion = torch.nn.HuberLoss()

        TradeExperience = namedtuple('TradeExperience', ('state', 'action', 'reward', 'nextState'))
        memoryBuffer = []
        memoryBufferCount = 0
        
        for epoch in range(self.epochs):

            print(f"Training DDQN - Current epoch {epoch+1}\r", end = "")
            
            modelPolicy.train()
            numberOfSharesOwned = np.zeros(self.numberOfStocks)
            currentBalance = self.initialCashBalance
            
            for dayNumber in range(len(self.encodedDataTrain)-1):

                state = self.getState(dayNumber, currentBalance, numberOfSharesOwned)

                with torch.no_grad():
                    qValues = modelPolicy(state)
                    qValues = qValues.view(self.numberOfStocks, 3)
                    actions = qValues.argmax(dim=1)
                actions = actions.cpu().numpy()
                
                randomActions = np.random.rand(self.numberOfStocks) < currentEps
                if randomActions.any():
                        actions[randomActions] = np.random.randint(0, 3, size=randomActions.sum())
                
                nextState, reward, currentBalance, numberOfSharesOwned = self.executeDiscreteActionTrain(actions, dayNumber, currentBalance, numberOfSharesOwned.copy())

                memory = TradeExperience(state.detach(), actions, reward, nextState.detach())
                
                if memoryBufferCount >= maxNumberForReplay:
                    memoryBuffer[memoryBufferCount%maxNumberForReplay] = memory
                else:
                    memoryBuffer.append(memory)
                
                memoryBufferCount += 1

                if (len(memoryBuffer) > minNumberForReplay) and (0 == (memoryBufferCount+1) % 5): # only sample every 5 steps
                    sampledMemories = random.sample(memoryBuffer, 32)

                    sampledMemoriesSplit = TradeExperience(*zip(*sampledMemories))

                    sampleStates = torch.stack(sampledMemoriesSplit.state).to(device)
                    sampleNextStates = torch.stack(sampledMemoriesSplit.nextState).to(device)
                    sampleActions = torch.tensor(np.array(sampledMemoriesSplit.action), dtype=torch.long, device=device)
                    sampleRewards = torch.tensor(np.array(sampledMemoriesSplit.reward), dtype=torch.float32, device=device)

                    # current q value
                    sampleQValues = modelPolicy(sampleStates).view(32, self.numberOfStocks, 3)
                    sampleQValues = sampleQValues.gather(2, sampleActions.unsqueeze(2)).squeeze(2)

                    # select action
                    with torch.no_grad():
                        nextActions = modelPolicy(sampleNextStates).view(32, self.numberOfStocks, 3)
                        nextActions = nextActions.argmax(dim=2, keepdim=True)

                        # now targets
                        nextTarget = modelTarget(sampleNextStates).view(32, self.numberOfStocks, 3)
                        nextStateValues = nextTarget.gather(2, nextActions).squeeze(2)

                    # bellman target
                    expectedQValues = sampleRewards.unsqueeze(1) + (bellmanCoefficient * nextStateValues)

                    loss = lossCriterion(sampleQValues, expectedQValues)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(modelPolicy.parameters(), max_norm=1.0)
                    optimizer.step()

            currentEps = max(epsDecay*currentEps, epsEnd)
            
            if 0 == ((epoch + 1) % 1): # reset target weights every epoch
                modelTarget.load_state_dict(modelPolicy.state_dict())
                modelTarget.eval()

            # run a small validation test
            if 0 == ((epoch+1) % 10):
                
                modelPolicy.eval()
                
                with torch.no_grad():

                    valBalance = self.initialCashBalance
                    valSharesOwned = np.zeros(self.numberOfStocks)
                    valReward = 0
                    returnValue = self.env.valReturnValue
                    minReturn = returnValue / (len(self.encodedDataValidation)-1) # harcoded returns for 3 years divided by number of val days

                    portfolioHistory = [1.0]
                    dailyReturns = []
                    
                    for valDayIndex in range(len(self.encodedDataValidation)-1):
                        valState = self.getStateVal(valDayIndex, valBalance, valSharesOwned)

                        valQValues = modelPolicy(valState)
                        valQValues = valQValues.view(self.numberOfStocks, 3)
                        valActions = valQValues.argmax(dim=1).cpu().numpy()

                        valReward, valBalance, valSharesOwned = self.executeDiscreteActionVal(valActions, valDayIndex, valBalance, valSharesOwned.copy())

                        netReturn = valReward - 1

                        prevMult = portfolioHistory[-1]
                        daysReturn = (valReward - prevMult) / prevMult
                        dailyReturns.append(daysReturn)
    
                        portfolioHistory.append(valReward)
                    
                    ROI = valReward

                    # Calmar Ratio
                    annualReturn = pow(ROI, 1/3)-1

                    portfolioHistory = np.array(portfolioHistory)
                    peaks = np.maximum.accumulate(portfolioHistory)
                    drawdowns = (peaks - portfolioHistory) / peaks
                    maxDrawdown = np.max(drawdowns)
                    calmarRatio = annualReturn / maxDrawdown if maxDrawdown > 0 else 0.0

                    # Sortino Ratio
                    dailyReturns = np.array(dailyReturns)
                    downsideReturns = dailyReturns[dailyReturns < minReturn]
                    totalValDays = len(self.encodedDataValidation) - 1
                    
                    if 0 < len(downsideReturns):
                        downsideDeviation = np.sqrt(np.mean((downsideReturns - minReturn) ** 2)) * np.sqrt(totalValDays) # avg number of trading days in a year
                        excessReturn = ROI - returnValue
                        sortinoRatio = excessReturn / downsideDeviation if downsideDeviation > 0 else 0.0
                    else:
                        sortinoRatio = 0.0

                    ratioValue = calmarRatio * sortinoRatio
                    
                    if (ROI > modelROI_Threshold):
                        if (ROI > highestROIAchieved):
                            highestROIAchieved = ROI
                            torch.save(modelPolicy.state_dict(), modelSavePathGreedy)

                        if (ratioValue > highestRatioAchieved):
                            highestRatioAchieved = ratioValue
                            torch.save(modelPolicy.state_dict(), modelSavePathBalanced)
                    
                    print(f"Epoch {epoch+1} | ROI {ROI} | Calmar {calmarRatio} | Sortino {sortinoRatio} - DDQN result over the validation set")

    # evaluates the model over the given year range. does not update the model. Tracks if model improves as it walks-forward through the test set.
    def testEval(self, modelPath, testEvalStartYearNumber, testEvalEndYearNumber):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # for the JSE: 34 stocks :       input : 69,                output: 102, dropout: 0.2, hidden layers 128, 128 and 128  
        modelPolicy = DeepNeuralNetwork(self.numberOfStocks*2 + 1, self.numberOfStocks*3, 0.2, 3, [128, 128, 128])
        policyWeights = torch.load(modelPath, weights_only=True)
        modelPolicy.load_state_dict(policyWeights)
        modelPolicy = modelPolicy.to(device)
        modelPolicy.eval()
        
        # evaluate the Performance on each test set year
        for testYear in range(testEvalStartYearNumber, testEvalEndYearNumber):
            with torch.no_grad():
                testBalance = self.initialCashBalance
                testSharesOwned = np.zeros(self.numberOfStocks)
                testReward = 0
                testYearDataStart = self.env.testYearIndex[testYear]
                if testYear >= self.env.numberOfTestYears -1:
                    testYearDataEnd = len(self.encodedDataTest)
                else:
                    testYearDataEnd = self.env.testYearIndex[testYear+1]
                    
                for testDayIndex in range(testYearDataStart, testYearDataEnd-1):
                    testState = self.getStateTest(testDayIndex, testBalance, testSharesOwned)
                    
                    testQValues = modelPolicy(testState)
                    testQValues = testQValues.view(self.numberOfStocks, 3)
                    testActions = testQValues.argmax(dim=1).cpu().numpy()

                    nextState, testReward, testBalance, testSharesOwned = self.executeDiscreteActionTest(testActions, testDayIndex, testBalance, testSharesOwned.copy())

                    if nextState == None:
                        print(f"Error: encountered error state on day {testDayIndex}, stopping evaluation")
                        continue

                ROI = testReward     
                print(f"Year: {testYear+2022} | ROI {ROI} - DDQN walk through result over the test set")

        del modelPolicy

    # evaluates the model for the given year while using the given shares and balance state. does not update the model.
    def testEvalRollingData(self, modelPath, testEvalYearNumber, testEvalBalance, testEvalSharesOwned): # same func as the test eval but we use rlling balance and stock data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # for the JSE: 34 stocks :       input : 69,                output: 102, dropout: 0.2, hidden layers 128, 128 and 128  
        modelPolicy = DeepNeuralNetwork(self.numberOfStocks*2 + 1, self.numberOfStocks*3, 0.2, 3, [128, 128, 128])
        policyWeights = torch.load(modelPath, weights_only=True)
        modelPolicy.load_state_dict(policyWeights)
        modelPolicy = modelPolicy.to(device)
        modelPolicy.eval()

        testYear = testEvalYearNumber
        
        # evaluate the Performance on each test set year
        with torch.no_grad():
            testBalance = testEvalBalance
            testSharesOwned = testEvalSharesOwned.copy()
            testReward = 0
            testYearDataStart = self.env.testYearIndex[testYear]
            if testYear >= self.env.numberOfTestYears -1:
                testYearDataEnd = len(self.encodedDataTest)
            else:
                testYearDataEnd = self.env.testYearIndex[testYear+1]
                    
            for testDayIndex in range(testYearDataStart, testYearDataEnd-1):
                testState = self.getStateTest(testDayIndex, testBalance, testSharesOwned)
                    
                testQValues = modelPolicy(testState)
                testQValues = testQValues.view(self.numberOfStocks, 3)
                testActions = testQValues.argmax(dim=1).cpu().numpy()

                nextState, testReward, testBalance, testSharesOwned = self.executeDiscreteActionTest(testActions, testDayIndex, testBalance, testSharesOwned.copy())

                if nextState == None:
                    print(f"Error: encountered error state on day {testDayIndex}, stopping evaluation")
                    continue

                prevMult = self.testPortfolioHistory[-1]
                daysReturn = (testReward - prevMult) / prevMult
                self.testReturns.append(daysReturn)
                self.testPortfolioHistory.append(testReward)
                self.shareCount.append(testSharesOwned.sum())

            ROI = testReward     
            print(f"Year: {testYear+2022} | ROI {ROI} - DDQN walk through result over the test set")

        del modelPolicy
        return testBalance, testSharesOwned

    # Executes walk-forward for the model for the given year. Updates the model and saves accordingly.
    def testTrain(self, modelPath, modelSavePath, walkEpochs, testYearIndex):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        learningRate = 0.000005
        
        weightDecay = 1e-4
        minNumberForReplay = 500
        maxNumberForReplay = 5000
        bellmanCoefficient = 0.99
        
        epsStart = 0.05
        epsEnd = 0.01
        epsDecay = 0.999
        currentEps = epsStart
        
        modelPolicy = DeepNeuralNetwork(self.numberOfStocks*2 + 1, self.numberOfStocks*3, 0.2, 3, [128, 128, 128])
        policyWeights = torch.load(modelPath, weights_only=True)
        modelPolicy.load_state_dict(policyWeights)
        modelPolicy = modelPolicy.to(device)
        modelPolicy.train()
        
        modelTarget = DeepNeuralNetwork(self.numberOfStocks*2 + 1, self.numberOfStocks*3, 0.2, 3, [128, 128, 128])
        modelTarget = modelTarget.to(device)
        modelTarget.load_state_dict(modelPolicy.state_dict())

        modelTarget.eval()
        
        optimizer = torch.optim.Adam(modelPolicy.parameters(), lr=learningRate, weight_decay=weightDecay)

        lossCriterion = torch.nn.HuberLoss()

        TradeExperience = namedtuple('TradeExperience', ('state', 'action', 'reward', 'nextState'))

        #print(sum(p.sum().item() for p in modelPolicy.parameters()))
        
        for epoch in range(walkEpochs):
            print(f"Walk-through Training DDQN - Current epoch {epoch+1}\r", end = "")
            
            modelPolicy.train()
            modelTarget.eval()
            
            numberOfSharesOwned = np.zeros(self.numberOfStocks)
            currentBalance = self.initialCashBalance

            testYearDataIndexStart = self.env.testYearIndex[testYearIndex]
            testYearDataIndexEnd = testYearDataIndexStart + 250
            if testYearIndex < self.env.numberOfTestYears - 1:
                testYearDataIndexEnd = self.env.testYearIndex[testYearIndex+1]
            
            for dayNumber in range(testYearDataIndexStart, testYearDataIndexEnd-1):
                state = self.getStateTest(dayNumber, currentBalance, numberOfSharesOwned)

                with torch.no_grad():
                    qValues = modelPolicy(state)
                    qValues = qValues.view(self.numberOfStocks, 3)
                    actions = qValues.argmax(dim=1)
                actions = actions.cpu().numpy()
                
                randomActions = np.random.rand(self.numberOfStocks) < currentEps
                if randomActions.any():
                        actions[randomActions] = np.random.randint(0, 3, size=randomActions.sum())
                
                nextState, reward, currentBalance, numberOfSharesOwned = self.executeDiscreteActionTest(actions, dayNumber, currentBalance, numberOfSharesOwned.copy())

                if nextState == None:
                    print(f"Error: encountered error state on day {testDayIndex}, stopping training")
                    continue
                
                memory = TradeExperience(state.detach(), actions, reward, nextState.detach())
                
                if self.rollingMemoryBufferCount >= maxNumberForReplay:
                    self.rollingMemoryBuffer[self.rollingMemoryBufferCount%maxNumberForReplay] = memory
                else:
                    self.rollingMemoryBuffer.append(memory)
                
                self.rollingMemoryBufferCount += 1

                if (len(self.rollingMemoryBuffer) > minNumberForReplay): # sample every step
                    sampledMemories = random.sample(self.rollingMemoryBuffer, 32)

                    sampledMemoriesSplit = TradeExperience(*zip(*sampledMemories))

                    sampleStates = torch.stack(sampledMemoriesSplit.state).to(device)
                    sampleNextStates = torch.stack(sampledMemoriesSplit.nextState).to(device)
                    sampleActions = torch.tensor(np.array(sampledMemoriesSplit.action), dtype=torch.long, device=device)
                    sampleRewards = torch.tensor(np.array(sampledMemoriesSplit.reward), dtype=torch.float32, device=device)

                    # current q value
                    sampleQValues = modelPolicy(sampleStates).view(32, self.numberOfStocks, 3)
                    sampleQValues = sampleQValues.gather(2, sampleActions.unsqueeze(2)).squeeze(2)

                    # select action
                    with torch.no_grad():
                        nextActions = modelPolicy(sampleNextStates).view(32, self.numberOfStocks, 3)
                        nextActions = nextActions.argmax(dim=2, keepdim=True)

                        # now targets
                        nextTarget = modelTarget(sampleNextStates).view(32, self.numberOfStocks, 3)
                        nextStateValues = nextTarget.gather(2, nextActions).squeeze(2)

                    # bellman target
                    expectedQValues = sampleRewards.unsqueeze(1) + (bellmanCoefficient * nextStateValues)

                    loss = lossCriterion(sampleQValues, expectedQValues)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(modelPolicy.parameters(), max_norm=1.0)
                    optimizer.step()

                if 0 == ((self.rollingMemoryBufferCount + 1) % 50): # reset target weights every 50 steps
                    modelTarget.load_state_dict(modelPolicy.state_dict())
                    modelTarget.eval()
            
            currentEps = max(epsDecay*currentEps, epsEnd)

        #print(sum(p.sum().item() for p in modelPolicy.parameters()))
        
        torch.save(modelPolicy.state_dict(), modelSavePath)

    # Manages walk-forward for the model for each year and the models produced.
    def testWalkThroughTrain(self, modelSavePath = 'DDQN_balanced.pth', env= None, walkEpochs = 10):
        if (not None == env):
            self.env = env

        self.rollingMemoryBuffer = []
        self.rollingMemoryBufferCount = 0

        currentModelUsed = modelSavePath

        for yearIndex in range(self.env.numberOfTestYears):
            break
            self.testEval(currentModelUsed, yearIndex, self.env.numberOfTestYears)
            if (yearIndex == self.env.numberOfTestYears-1):
                break
            currentModelSavePath = modelSavePath[0:-4] + f"_walk-through_year-{yearIndex}.pth"
            self.testTrain(currentModelUsed, currentModelSavePath, walkEpochs=walkEpochs, testYearIndex = yearIndex)
            currentModelUsed = currentModelSavePath

        balance = self.initialCashBalance
        numberOfSharesOwned = np.zeros(self.numberOfStocks)

        currentModelUsed = modelSavePath

        #print("")

        self.testPortfolioHistory = [1.0]
        self.testReturns = []
        self.shareCount = [0]
        
        for yearIndex in range(self.env.numberOfTestYears):
            balance, numberOfSharesOwned = self.testEvalRollingData(currentModelUsed, yearIndex, balance, numberOfSharesOwned.copy())
            currentModelSavePath = modelSavePath[0:-4] + f"_walk-through_year-{yearIndex}.pth"
            currentModelUsed = currentModelSavePath

        #print(self.testPortfolioHistory)
        #print(self.tradeCount)
        #print("Shares counts are:")
        #print(self.shareCount)
        
        convertedNextDayOpenPrices = np.nan_to_num(self.env.testOpenPrices[-1], nan=0.0)
        totalPortfolioValue = balance + np.sum(numberOfSharesOwned * convertedNextDayOpenPrices)
        ROI = (totalPortfolioValue/self.initialCashBalance)

        annualReturn = pow(ROI, 1/4)-1
        totalNumberOfTestDays = len(self.encodedDataTest)
        daysPerYear = totalNumberOfTestDays/4

        # Calmar ratio
        portfolioHistory = np.array(self.testPortfolioHistory)
        peaks = np.maximum.accumulate(self.testPortfolioHistory)
        drawdowns = (peaks - self.testPortfolioHistory) / peaks
        maxDrawdown = np.max(drawdowns)
        calmarRatio = annualReturn / maxDrawdown if maxDrawdown > 0 else 0.0

        # Sortino ratio
        # we use the index as the MAR which we have stored hardcoded values in the environment.
        minReturns = []
        for dayIndex in range(len(self.encodedDataTest)):
            if dayIndex < self.env.testYearIndex[1]:
                if dayIndex == self.env.testYearIndex[0]:
                    continue
                minReturns.append((self.env.testReturnValue[0]-1)/(self.env.testYearIndex[1] - self.env.testYearIndex[0]))
            elif dayIndex < self.env.testYearIndex[2]:
                if dayIndex == self.env.testYearIndex[1]:
                    continue
                minReturns.append((self.env.testReturnValue[1]-1)/(self.env.testYearIndex[2] - self.env.testYearIndex[1]))
            elif dayIndex < self.env.testYearIndex[3]:
                if dayIndex == self.env.testYearIndex[2]:
                    continue
                minReturns.append((self.env.testReturnValue[2]-1)/(self.env.testYearIndex[3] - self.env.testYearIndex[2]))
            else:
                if dayIndex == self.env.testYearIndex[3]:
                    continue
                minReturns.append((self.env.testReturnValue[3]-1)/(len(self.encodedDataTest) - self.env.testYearIndex[3]))
        minReturns = np.array(minReturns)
        dailyReturns = np.array(self.testReturns)

        downsideMask = dailyReturns < minReturns
        downsideReturns = dailyReturns[downsideMask]
        downsideTargets = minReturns[downsideMask]
                    
        if 0 < len(downsideReturns):
            downsideDeviation = np.sqrt(np.mean((downsideReturns - downsideTargets) ** 2)) * np.sqrt(daysPerYear) # hardcoded to number of days in a year
            excessReturn = annualReturn - np.mean(minReturns) * daysPerYear
            sortinoRatio = excessReturn / downsideDeviation if downsideDeviation > 0 else 0.0
        else:
            sortinoRatio = 0.0
        
        print(f"Final ROI: {ROI:.3f} | Calmar: {calmarRatio:.3f} | Sortino: {sortinoRatio:.3f}")
        