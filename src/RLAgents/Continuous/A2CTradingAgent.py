from src.RLAgents.Continuous.ContinuousTradingAgent import ContinuousTradingAgent
from src.ActorDNN import ActorDNN
from src.DeepNeuralNetwork import DeepNeuralNetwork
from src.RLAgents.RLTradingAgent import EncoderCode
from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np

# A2C class
class A2CTradingAgent(ContinuousTradingAgent):
    def __init__(self, env, encoderModel, encoderCode = EncoderCode.DNN.value, initialCashBalance = 1000000, epochs = 500):
        super().__init__(env, initialCashBalance, encoderModel, encoderCode = encoderCode)
        self.epochs = epochs

    # trains model over the train set and saves based on val set performance
    def train(self, modelROI_Threshold = 0, modelSavePathGreedy = 'A2C_greedy.pth', modelSavePathBalanced = 'A2C_balanaced.pth',
              modelSavePathGreedyCritic = 'A2C_greedyCritic.pth', modelSavePathBalancedCritic = 'A2C_balancedCritic.pth', env = None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if (not None == env):
            self.env = env
        
        learningRate = 0.0005
        weightDecay = 1e-5
        bellmanCoefficient = 0.99

        trajectoryLength = 64

        highestRatioAchieved = -10
        highestROIAchieved = 0
        
        # actor  
        modelActor = ActorDNN(self.numberOfStocks*2 + 1, self.numberOfStocks, 0.2, 3, [128, 128, 64], self.numberOfStocks)
        modelActor = modelActor.to(device)
        
        # critic
        modelCritic = DeepNeuralNetwork(self.numberOfStocks*2 + 1, 1, 0.2, 3, [128, 64, 32])
        modelCritic = modelCritic.to(device)

        optimizer = torch.optim.Adam(list(modelActor.parameters()) + list(modelCritic.parameters()), lr=learningRate, weight_decay=weightDecay)
        lossCriterion = torch.nn.HuberLoss()
        
        TradingDayLoggedData = namedtuple('TradingDayLoggedData', ('state', 'action', 'reward', 'stateValue'))
        
        for epoch in range(self.epochs):

            print(f"Training A2C - Current epoch {epoch+1}\r", end = "")
            
            modelActor.train()
            modelCritic.train()

            trajectoryBuffer = []
            
            numberOfSharesOwned = np.zeros(self.numberOfStocks)
            currentBalance = self.initialCashBalance
            
            for dayNumber in range(len(self.encodedDataTrain)-1):

                # features - 69 for JSE
                state = self.getState(dayNumber, currentBalance, numberOfSharesOwned)

                # actions is actions per stock
                with torch.no_grad():
                    means, stdDev = modelActor(state)
                    stateValue = modelCritic(state)

                    normalDistribution = torch.distributions.Normal(means.squeeze(0), stdDev.squeeze(0))
                    actions = normalDistribution.sample()

                actions = actions.cpu().numpy()
                actions = np.clip(actions, -1.0, 1.0)
                
                reward, currentBalance, numberOfSharesOwned = self.executeContinuousActionTrain(actions, dayNumber, currentBalance, numberOfSharesOwned.copy())
                
                memory = TradingDayLoggedData(state.detach(), actions.copy(), reward, stateValue.item())
                trajectoryBuffer.append(memory)
                
                if (len(trajectoryBuffer) >= trajectoryLength) or (dayNumber == len(self.encodedDataTrain)-2):

                    bufferStates = torch.stack([m.state for m in trajectoryBuffer]).to(device)
                    bufferActions = torch.tensor(np.array([m.action for m in trajectoryBuffer]), dtype=torch.float32, device=device)
                    bufferRewards = np.array([m.reward for m in trajectoryBuffer])
                
                    bufferMeans, bufferStdDev = modelActor(bufferStates)
                    bufferStateValues = modelCritic(bufferStates).flatten()
                
                    normDist = torch.distributions.Normal(bufferMeans, bufferStdDev)
                    logProbs = normDist.log_prob(bufferActions).sum(dim=-1)
                    
                    with torch.no_grad():
                        nextState = self.getState(dayNumber+1, currentBalance, numberOfSharesOwned.copy())
                        nextValue = modelCritic(nextState).item()
                    
                    targets = []
                    
                    runningTarget = nextValue

                    for r in reversed(bufferRewards):
                        runningTarget = r + bellmanCoefficient * runningTarget
                        targets.insert(0, runningTarget)

                    targets = torch.tensor(targets, dtype=torch.float32, device=device)
                    advantages = targets - bufferStateValues

                    actorLoss = (-logProbs.flatten() * advantages.detach()).sum()
                    criticLoss = lossCriterion(bufferStateValues, targets.detach())
                    loss = actorLoss + criticLoss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(modelActor.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(modelCritic.parameters(), max_norm=1.0)
                    optimizer.step()

                    trajectoryBuffer.clear()
            
            if 0 == ((epoch + 1) % 10):
                modelActor.eval()
                
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

                        valMeans, valStdDev = modelActor(valState)
                        valActions = np.clip(valMeans.squeeze(0).cpu().numpy(), -1.0, 1.0)

                        valReward, valBalance, valSharesOwned = self.executeContinuousActionVal(valActions, valDayIndex, valBalance, valSharesOwned.copy())

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
                            torch.save(modelActor.state_dict(), modelSavePathGreedy)
                            torch.save(modelCritic.state_dict(), modelSavePathGreedyCritic)

                        if (ratioValue > highestRatioAchieved):
                            highestRatioAchieved = ratioValue
                            torch.save(modelActor.state_dict(), modelSavePathBalanced)
                            torch.save(modelCritic.state_dict(), modelSavePathBalancedCritic)
                    
                    print(f"Epoch {epoch+1} | ROI {ROI} | Calmar {calmarRatio} | Sortino {sortinoRatio} - A2C result over the validation set")

    # evaluates the model over the given year range. does not update the model. Tracks if model improves as it walks-forward through the test set.
    def testEval(self, modelPathActor, testEvalStartYearNumber, testEvalEndYearNumber):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        modelActor = ActorDNN(self.numberOfStocks*2 + 1, self.numberOfStocks, 0.2, 3, [128, 128, 64], self.numberOfStocks)
        modelActor = modelActor.to(device)
        actorWeights = torch.load(modelPathActor, weights_only=True)
        modelActor.load_state_dict(actorWeights)
        modelActor.eval()
    
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

                    testMeans, testStdDev = modelActor(testState)
                    testActions = np.clip(testMeans.squeeze(0).cpu().numpy(), -1.0, 1.0)

                    testReward, testBalance, testSharesOwned = self.executeContinuousActionTest(testActions, testDayIndex, testBalance, testSharesOwned.copy())

                ROI = testReward
                print(f"Year: {testYear+2022} | ROI {ROI} - A2C walk through result over the test set")

        del modelActor

    # evaluates the model for the given year while using the given shares and balance state. does not update the model.
    def testEvalRollingData(self, modelActorPath, testEvalYearNumber, testEvalBalance, testEvalSharesOwned): 
        # same func as the test eval but we use a rolling balance and shares owned
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # for the JSE: 34 stocks :       input : 69,                output: 102, dropout: 0.2, hidden layers 128, 128 and 128  
        modelActor = ActorDNN(self.numberOfStocks*2 + 1, self.numberOfStocks, 0.2, 3, [128, 128, 64], self.numberOfStocks)
        modelActor = modelActor.to(device)
        actorWeights = torch.load(modelActorPath, weights_only=True)
        modelActor.load_state_dict(actorWeights)
        modelActor.eval()

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

                testMeans, testStdDev = modelActor(testState)
                testActions = np.clip(testMeans.squeeze(0).cpu().numpy(), -1.0, 1.0)

                testReward, testBalance, testSharesOwned = self.executeContinuousActionTest(testActions, testDayIndex, testBalance, testSharesOwned.copy())

                prevMult = self.testPortfolioHistory[-1]
                daysReturn = (testReward - prevMult) / prevMult
                self.testReturns.append(daysReturn)
                self.testPortfolioHistory.append(testReward)

                self.shareCount.append(testSharesOwned.sum())

            ROI = testReward     
            #print(f"Year: {testYear+2022} | ROI {ROI} - A2C walk through result over the test set")

        del modelActor
        return testBalance, testSharesOwned

    # Executes walk-forward for the model for the given year. Updates the model and saves accordingly.
    def testTrain(self, modelActorPath, modelCriticPath, modelActorSavePath, modelCriticSavePath, walkEpochs, testYearIndex):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        learningRate = 0.0005
        
        weightDecay = 1e-5
        bellmanCoefficient = 0.99
        
        trajectoryLength = 64

        entropyCoefficient = 0.05
        
        # actor  
        modelActor = ActorDNN(self.numberOfStocks*2 + 1, self.numberOfStocks, 0.2, 3, [128, 128, 64], self.numberOfStocks)
        actorWeights = torch.load(modelActorPath, weights_only=True)
        modelActor.load_state_dict(actorWeights)
        modelActor = modelActor.to(device)
        
        # critic
        modelCritic = DeepNeuralNetwork(self.numberOfStocks*2 + 1, 1, 0.2, 3, [128, 64, 32])
        criticWeights = torch.load(modelCriticPath, weights_only=True)
        modelCritic.load_state_dict(criticWeights)
        modelCritic = modelCritic.to(device)

        optimizer = torch.optim.Adam(list(modelActor.parameters()) + list(modelCritic.parameters()), lr=learningRate, weight_decay=weightDecay)
        lossCriterion = torch.nn.HuberLoss()
        
        TradingDayLoggedData = namedtuple('TradingDayLoggedData', ('state', 'action', 'reward', 'stateValue'))

        #print(sum(p.sum().item() for p in modelActor.parameters()))
        #print(sum(p.sum().item() for p in modelCritic.parameters()))
        
        for epoch in range(walkEpochs):
            print(f"Walk-through Training A2C - Current epoch {epoch+1}\r", end = "")
            
            modelActor.train()
            modelCritic.train()

            trajectoryBuffer = []
            
            numberOfSharesOwned = np.zeros(self.numberOfStocks)
            currentBalance = self.initialCashBalance

            testYearDataIndexStart = self.env.testYearIndex[testYearIndex]
            testYearDataIndexEnd = testYearDataIndexStart + 250
            if testYearIndex < self.env.numberOfTestYears - 1:
                testYearDataIndexEnd = self.env.testYearIndex[testYearIndex+1]
            
            for dayNumber in range(testYearDataIndexStart, testYearDataIndexEnd-1):
                # features - 69 for JSE
                state = self.getStateTest(dayNumber, currentBalance, numberOfSharesOwned)

                # actions is actions per stock
                with torch.no_grad():
                    means, stdDev = modelActor(state)
                    stateValue = modelCritic(state)

                    normalDistribution = torch.distributions.Normal(means.squeeze(0), stdDev.squeeze(0))
                    actions = normalDistribution.sample()

                actions = actions.cpu().numpy()
                actions = np.clip(actions, -1.0, 1.0)
                
                reward, currentBalance, numberOfSharesOwned = self.executeContinuousActionTest(actions, dayNumber, currentBalance, numberOfSharesOwned.copy())
                
                memory = TradingDayLoggedData(state.detach(), actions.copy(), reward, stateValue.item())
                trajectoryBuffer.append(memory)
                
                if (len(trajectoryBuffer) >= trajectoryLength) or (dayNumber == len(self.encodedDataTrain)-2):

                    bufferStates = torch.stack([m.state for m in trajectoryBuffer]).to(device)
                    bufferActions = torch.tensor(np.array([m.action for m in trajectoryBuffer]), dtype=torch.float32, device=device)
                    bufferRewards = np.array([m.reward for m in trajectoryBuffer])
                
                    bufferMeans, bufferStdDev = modelActor(bufferStates)
                    bufferStateValues = modelCritic(bufferStates).flatten()
                
                    normDist = torch.distributions.Normal(bufferMeans, bufferStdDev)
                    logProbs = normDist.log_prob(bufferActions).sum(dim=-1)
                    
                    with torch.no_grad():
                        nextState = self.getStateTest(dayNumber+1, currentBalance, numberOfSharesOwned.copy())
                        nextValue = modelCritic(nextState).item()
                    
                    targets = []
                    
                    runningTarget = nextValue

                    for r in reversed(bufferRewards):
                        runningTarget = r + bellmanCoefficient * runningTarget
                        targets.insert(0, runningTarget)

                    targets = torch.tensor(targets, dtype=torch.float32, device=device)
                    advantages = targets - bufferStateValues

                    entropy = normDist.entropy().sum(dim=-1).mean()
                    
                    actorLoss = (-logProbs.flatten() * advantages.detach()).sum() - (entropyCoefficient * entropy)
                    criticLoss = lossCriterion(bufferStateValues, targets.detach())
                    loss = actorLoss + criticLoss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(modelActor.parameters(), max_norm=2.0)
                    torch.nn.utils.clip_grad_norm_(modelCritic.parameters(), max_norm=2.0)
                    optimizer.step()

                    trajectoryBuffer.clear()

        #print("")
        #print(sum(p.sum().item() for p in modelActor.parameters()))
        #print(sum(p.sum().item() for p in modelCritic.parameters()))
        
        torch.save(modelActor.state_dict(), modelActorSavePath)
        torch.save(modelCritic.state_dict(), modelCriticSavePath)

    # Manages walk-forward for the model for each year and the models produced.
    def testWalkThroughTrain(self, modelActorPath = 'A2C_balanced_actor.pth', modelCriticPath = 'A2C_balanced_actor.pth', env= None, walkEpochs = 10):
        if (not None == env):
            self.env = env

        self.rollingMemoryBuffer = []
        self.rollingMemoryBufferCount = 0

        currentActorModelUsed = modelActorPath
        currentCriticModelUsed = modelCriticPath

        for yearIndex in range(self.env.numberOfTestYears):
            self.testEval(currentActorModelUsed, yearIndex, self.env.numberOfTestYears)
            if (yearIndex == self.env.numberOfTestYears-1):
                break
            
            currentActorModelSavePath = modelActorPath[0:-4] + f"_walk-through_year-{yearIndex}.pth"
            currentCriticModelSavePath = modelCriticPath[0:-4] + f"_walk-through_year-{yearIndex}.pth"
            
            self.testTrain(currentActorModelUsed, currentCriticModelUsed, currentActorModelSavePath, currentCriticModelSavePath, walkEpochs=walkEpochs, testYearIndex = yearIndex)
            
            currentActorModelUsed = currentActorModelSavePath
            currentCriticModelUsed = currentCriticModelSavePath

        balance = self.initialCashBalance
        numberOfSharesOwned = np.zeros(self.numberOfStocks)

        currentActorModelUsed = modelActorPath

        print("")

        self.testPortfolioHistory = [1.0]
        self.testReturns = []
        self.shareCount = [0]
        
        for yearIndex in range(self.env.numberOfTestYears):
            balance, numberOfSharesOwned = self.testEvalRollingData(currentActorModelUsed, yearIndex, balance, numberOfSharesOwned.copy())
            currentActorModelSavePath = modelActorPath[0:-4] + f"_walk-through_year-{yearIndex}.pth"
            currentActorModelUsed = currentActorModelSavePath

        #print(self.testPortfolioHistory)
        #print("Total Shares", self.tradeCount)
        #print("Buy", self.tradeCountBuy)
        #print("Sell", self.tradeCountSell)

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
        
        #print(f"Final ROI: {ROI:.3f} | Calmar: {calmarRatio:.3f} | Sortino: {sortinoRatio:.3f}")

if __name__ == "__main__":
    A2CTradingAgent= A2CTradingAgent()
    A2CTradingAgent.train()