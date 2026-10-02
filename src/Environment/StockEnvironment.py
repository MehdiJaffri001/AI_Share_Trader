import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import random
import tsl
from tsl.data import SpatioTemporalDataset
from tsl.data.loader import StaticGraphLoader

def seedWorker(workerID):
    # Ensures data workers use the exact same reference seed boundaries
    workerSeed = torch.initial_seed() % 2**32
    np.random.seed(workerSeed)
    random.seed(workerSeed)

# extracts raw data from datasets and processes them into loaders for encoders.
class StockEnvironment:
    def __init__(self, useJSE = True, flattenedNodes = True):

        # used hardcoded return values obtained by simple calculation or searching online.  
        if useJSE:
            JSE_DataRaw = pd.read_excel('data/rawData/StockData_JSA_&_NYSE.xlsx', sheet_name='Top40 Index', parse_dates=['Date'])
            data = JSE_DataRaw.rename(columns={JSE_DataRaw.columns[0]: 'Token'})
            self.valReturnValue = 1.44
            self.testReturnValue = [0.9986, 1.0529, 1.0693, 1.4324]
        else:
            NYSE_DataRaw = pd.read_excel('data/rawData/StockData_JSA_&_NYSE.xlsx', sheet_name='Dow Jones Index', parse_dates=['Date'])
            data = NYSE_DataRaw.rename(columns={NYSE_DataRaw.columns[0]: 'Token'})
            self.valReturnValue = 1.57
            self.testReturnValue = [0.9122, 1.1370, 1.1288, 1.1297]

        # Clean Data
        data.dropna(subset=['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume', 'Total Return'], inplace=True)

        # Cuttoff stocks without enough data
        trainingCutoff = pd.to_datetime('2016-01-01')
        earliestTradeDates = data.groupby('Token')['Date'].min()
        validStocks = earliestTradeDates[earliestTradeDates <= trainingCutoff].index

        data = data[data['Token'].isin(validStocks)]
        data = data.drop_duplicates(subset=['Date', 'Token'], keep='last')
        data = data.sort_values(by=['Date', 'Token'], ascending=True)

        # reorganize data into 3d [number of days][number of stocks][features]
        pivotDF = data.pivot(index='Date', columns='Token', values=['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume', 'Total Return'])
        dates = pivotDF.index
        tokens = pivotDF.columns.get_level_values('Token').unique()

        #print(tokens)
        
        # extract raw features
        openPrice = pivotDF['Open Price']
        closePrice = pivotDF['Close Price']
        highPrice = pivotDF['High Price']
        lowPrice = pivotDF['Low Price']
        volume = pivotDF['Volume']
        returnPrice = pivotDF['Total Return']/100
        
        # convert features into relative daily changes
        openPriceRelChange = (openPrice-closePrice.shift(1))/(closePrice.shift(1)+ 1e-8)
        prevDayRelReturn = returnPrice.shift(1)
        highRelChange = (highPrice.shift(1)-openPrice.shift(1))/(openPrice.shift(1)+ 1e-8)
        lowRelChange = (lowPrice.shift(1)-openPrice.shift(1))/(openPrice.shift(1)+ 1e-8)

        # keeps track of currently traded volume using a weighted volume where older values decay
        volumeEWMA = volume.ewm(alpha=0.2, adjust=False).mean()
        volumeRelChange = volume.shift(1) / (volumeEWMA.shift(1) + 1e-8)

        # Garman-Klass volitility as a feature - shows stock volatility during day
        logHL = np.log(highPrice.shift(1) / lowPrice.shift(1))
        logCO = np.log(closePrice.shift(1) / openPrice.shift(1))
        GarmanKlassVolatility = np.sqrt(np.clip(0.5 * (logHL ** 2) - (2 * np.log(2) - 1) * (logCO ** 2), 0, None))

        # Price volatility over time with decay.
        volatilityEWMA = prevDayRelReturn.ewm(alpha=0.06, adjust=False).std()

        isInTrainSet = openPriceRelChange.index < pd.to_datetime('2019-01-01')

        # Normalize data based on train set
        for df in [openPriceRelChange, prevDayRelReturn, highRelChange, lowRelChange, volumeRelChange]:
            mean = df[isInTrainSet].mean()
            std = df[isInTrainSet].std() + 1e-8
            df.update((df - mean) / std)

        gkMin, gkMax = GarmanKlassVolatility[isInTrainSet].min().min(), GarmanKlassVolatility[isInTrainSet].max().max()
        GarmanKlassVolatility.update((GarmanKlassVolatility - gkMin) / (gkMax - gkMin + 1e-8))

        ewmaVolMin, ewmaVolMax = volatilityEWMA[isInTrainSet].min().min(), volatilityEWMA[isInTrainSet].max().max()
        volatilityEWMA.update((volatilityEWMA - ewmaVolMin) / (ewmaVolMax - ewmaVolMin + 1e-8))

        # availability mask
        totalMask = pivotDF['Open Price'].notna() & pivotDF['High Price'].notna() & pivotDF['Low Price'].notna() & pivotDF['Close Price'].notna() & pivotDF['Volume'].notna() & pivotDF['Total Return'].notna()

        availabilityMask = totalMask.astype(float).values

        featureList = [openPriceRelChange, prevDayRelReturn, highRelChange, lowRelChange, volumeRelChange, GarmanKlassVolatility, volatilityEWMA]
        for df in featureList:
            df.fillna(-99.0, inplace=True)

        # converts data into 3d array [days][stocks][features]
        processedData = np.stack([df.values for df in featureList], axis=-1)

        targets = returnPrice.fillna(0.0).values
        
        # split data
        trainStartIndex = 6 # Skips first few days to allow for stable features
        validationStartIndex = np.searchsorted(dates, pd.to_datetime('2019-01-01'))
        testStartIndex  = np.searchsorted(dates, pd.to_datetime('2022-01-01'))
        
        self.trainOpenPrices = openPrice.values[trainStartIndex:validationStartIndex]
        self.valOpenPrices = openPrice.values[validationStartIndex:testStartIndex]
        self.testOpenPrices = openPrice.values[testStartIndex:]

        self.trainClosePrices = closePrice.values[trainStartIndex:validationStartIndex]
        self.valClosePrices = closePrice.values[validationStartIndex:testStartIndex]
        self.testClosePrices = closePrice.values[testStartIndex:]
        
        self.numberOfStocks = len(tokens)
        self.numberOfTimesteps = len(dates) - trainStartIndex
        numberOfFeatures = 7
        self.flattenedNumberOfFeatures = self.numberOfStocks * numberOfFeatures
        
        trainData = processedData[trainStartIndex:validationStartIndex]
        valData   = processedData[validationStartIndex:testStartIndex]
        testData  = processedData[testStartIndex:]

        startIndex = np.searchsorted(dates, pd.to_datetime('2022-01-01'))
        self.testYearIndex = [
            0, 
            np.searchsorted(dates, pd.to_datetime('2023-01-01')) - startIndex,
            np.searchsorted(dates, pd.to_datetime('2024-01-01')) - startIndex,
            np.searchsorted(dates, pd.to_datetime('2025-01-01')) - startIndex
        ]

        #print(f"2022 is {0} to {self.testYearIndex[1]}")
        #print(f"2023 is {self.testYearIndex[1]} to {self.testYearIndex[2]}")
        #print(f"2024 is {self.testYearIndex[2]} to {self.testYearIndex[3]}")
        #print(f"2025 is {self.testYearIndex[3]} to {len(self.testOpenPrices)}")
        
        self.numberOfTestYears = 4

        # fomat data for DNN or STGNN respectively.
        if flattenedNodes:
            trainData = trainData.reshape(trainData.shape[0], self.flattenedNumberOfFeatures)
            valData   = valData.reshape(valData.shape[0], self.flattenedNumberOfFeatures)
            testData  = testData.reshape(testData.shape[0], self.flattenedNumberOfFeatures)            
        else:
            targets = np.expand_dims(targets, axis = -1)
            availabilityMask = np.expand_dims(availabilityMask, axis = -1)
            availabilityMask = availabilityMask.astype(bool)
            
        trainTargets = targets[trainStartIndex:validationStartIndex]
        valTargets = targets[validationStartIndex:testStartIndex]
        testTargets = targets[testStartIndex:]

        trainDates = dates[trainStartIndex:validationStartIndex]
        valDates = dates[validationStartIndex:testStartIndex]
        testDates = dates[testStartIndex:]        

        trainMask = availabilityMask[trainStartIndex:validationStartIndex]
        valMask = availabilityMask[validationStartIndex:testStartIndex]
        testMask = availabilityMask[testStartIndex:]

        # fomat datasets for DNN or STGNN respectively.
        if flattenedNodes:
            self.trainDataset = TensorDataset(
                torch.tensor(trainData, dtype=torch.float32),
                torch.tensor(trainTargets, dtype=torch.float32),
                torch.tensor(trainMask, dtype=torch.float32)
            )

            self.valDataset = TensorDataset(
                torch.tensor(valData, dtype=torch.float32),
                torch.tensor(valTargets, dtype=torch.float32),
                torch.tensor(valMask, dtype=torch.float32)
            )

            self.testDataset = TensorDataset(
                torch.tensor(testData, dtype=torch.float32),
                torch.tensor(testTargets, dtype=torch.float32),
                torch.tensor(testMask, dtype=torch.float32)
            )
        else:

            windowLength = 20
            horizonLength = 1
            strideLength = 1
            
            self.trainDataset = SpatioTemporalDataset(
                target=trainTargets.copy(),                                                              # Target shape is [days, stocks, 1]
                mask = trainMask.copy(),
                covariates={
                    'u': trainData.copy(),                                                        # Feature shape is [days, stocks, 7]
                },
                input_map={'x': ['target', 'u']}, 
                window=windowLength,                                                               # historical window is set to 20, should be around 1 month 
                horizon=horizonLength,                                                                        # Predict next day returns
                stride=strideLength
            )

            self.valDataset = SpatioTemporalDataset(
                target=valTargets.copy(),
                mask = valMask.copy(),
                covariates={
                    'u': valData.copy(),
                },
                input_map={'x': ['target', 'u']},
                window=windowLength,
                horizon=horizonLength,
                stride=strideLength
            )

            self.testDataset = SpatioTemporalDataset(
                target=testTargets.copy(),
                mask = testMask.copy(),
                covariates={
                    'u': testData.copy(),
                },
                input_map={'x': ['target', 'u']}, 
                window=windowLength,
                horizon=horizonLength,
                stride=strideLength
            )
        
        # random Gen for seeding
        self.gen = torch.Generator()

        # default Loader, other uses require calling the function with different configurations after init
        self.createLoader()
    
    def seedEnv(self, seed):
        self.gen.manual_seed(seed)
    
    def createLoader(self, shuffleTrain = True, batchSize = 32, formatStaticGraph = False):
        # create loaders for DNN or STGNN respectively.
        if formatStaticGraph:
            self.trainLoader = StaticGraphLoader(self.trainDataset, batch_size=batchSize, shuffle=shuffleTrain, worker_init_fn=seedWorker, generator=self.gen,
                                                 pin_memory=True)
            self.valLoader = StaticGraphLoader(self.valDataset, batch_size=batchSize, shuffle=False, worker_init_fn=seedWorker, generator=self.gen, 
                                               pin_memory=True)
            self.testLoader = StaticGraphLoader(self.testDataset, batch_size=batchSize, shuffle=False, worker_init_fn=seedWorker, generator=self.gen,
                                                pin_memory=True)

        else:
            self.trainLoader = DataLoader(self.trainDataset, batch_size = batchSize, shuffle=shuffleTrain, worker_init_fn=seedWorker, generator=self.gen)
            self.valLoader   = DataLoader(self.valDataset, batch_size = batchSize, shuffle=False, worker_init_fn=seedWorker, generator=self.gen)
            self.testLoader = DataLoader(self.testDataset, batch_size = batchSize, shuffle=False, worker_init_fn=seedWorker, generator=self.gen)

        # transaction fee of 0.25% of whatever we spend, which we consider suitable for both markets.
        self.transactionFee = 0.0025

if __name__ == "__main__":
    flattened = False
    env = StockEnvironment(useJSE = True, flattenedNodes = flattened)
    if flattened:
        env.createLoader()
        print(env.trainLoader.dataset.tensors[0].shape)
        print(env.trainLoader.dataset.tensors[1].shape)
        print(env.trainLoader.dataset.tensors[2].shape)
    
        print(env.valLoader.dataset.tensors[0].shape)
        print(env.valLoader.dataset.tensors[1].shape)
        print(env.valLoader.dataset.tensors[2].shape)
    
        print(env.testLoader.dataset.tensors[0].shape)
        print(env.testLoader.dataset.tensors[1].shape)
        print(env.testLoader.dataset.tensors[2].shape)
    else:
        env.createLoader(shuffleTrain = True, batchSize = 32, formatStaticGraph = True)
        print(env.trainLoader.dataset.shape)
        print(env.trainLoader.dataset.features.shape)
        print(env.valLoader.dataset.shape)
        print(env.testLoader.dataset.shape)
