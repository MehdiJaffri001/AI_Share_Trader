# this py file is designed to run the entire project. 
# the trainModels func trains all the encoders and then the models on the designated train data 
# the testWalkThrough func executes the walk-forward testing on the models. 

import sys

seeds = [41, 101, 777, 999, 2026]
encoders = ["DNN", "GWN"]
rlAgents = ["PPO", "A2C", "DDQN"]
globalUseJSE = True # defualt to JSE
if len(sys.argv)>1:
    if sys.argv[1] == "JSE":
        globalUseJSE = True
    elif sys.argv[1] == "NYSE":
        globalUseJSE = False

from src.Environment.StockEnvironment import StockEnvironment
from src.Encoders.EncoderGenerator import EncoderGenerator
from src.Encoders.STGNN_EncoderGen import STGNNEncoderGenerator
from src.RLAgents.Discrete.DDQNTradingAgent import DDQNTradingAgent
from src.RLAgents.Continuous.A2CTradingAgent import A2CTradingAgent
from src.RLAgents.Continuous.PPOTradingAgent import PPOTradingAgent
from src.RLAgents.RLTradingAgent import EncoderCode
from src.DeepNeuralNetwork import DeepNeuralNetwork

import gc
import random
import torch
import numpy as np

from tsl.nn.models import GraphWaveNetModel
from tsl.nn.models import AGCRNModel

env = StockEnvironment(useJSE = globalUseJSE, flattenedNodes = True)
baselineEncoderModel = DeepNeuralNetwork(env.flattenedNumberOfFeatures, env.numberOfStocks, 0.10, 3, [256, 128, 64])
graphWavenetEncoderModel = GraphWaveNetModel(
            input_size = 8, # hardcoded number of features + 1
            output_size = 1,
            horizon = 1,
            n_nodes = env.numberOfStocks,
            learned_adjacency = True,
            emb_size = 12,
            hidden_size = 32
        )
agcrnEncoderModel = AGCRNModel(
    input_size = 8,
    output_size = 1,
    horizon = 1,
    n_nodes = env.numberOfStocks,
    emb_size = 12,
    hidden_size = 32,
    n_layers = 1
)

encoderDNNPath = 'DNNEncoder_stable.pth'
encoderGWNPath = 'GWNEncoder_stable.pth'
encoderAGCRNPath = 'AGCRNEncoder_stable.pth'

# functions are created to make the system modular and reduce cognitive overhead whilst reducing code length.
# In the case that the whole suite is not needed, specific line may be commented before running essentially disabling them.

def loadDNNEncoderWeights(savePath):
    baselineEncoderModel.load_state_dict(torch.load(savePath + encoderDNNPath))

def loadGWNEncoderWeights(savePath):
    graphWavenetEncoderModel.load_state_dict(torch.load(savePath+encoderGWNPath))

def loadAGCRNEncoderWeights(savePath):
    agcrnEncoderModel.load_state_dict(torch.load(savePath+encoderAGCRNPath))

# Sets the seed before each model trains and ensures the reproducibility of the project.
# Running the file on the same device results in the same results everytime, but output may differ based on architecture, pytorch version and potential GPU quirks.
def setSeed(seed):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    env.seedEnv(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    print(f"Using seed: {seed} ")

# generates the dnn baselne encoder
def genDNNEncoder(epochs, env, savePath):
    encoderGenerator = EncoderGenerator(numberOfEpochs = epochs, env = env, savePath = savePath)
    encoderGenerator.generateModels()

# generates the gwn encoder or the agcrn encoder based on the code given.
def genSTGNNEncoder(epochs, env, savePath, encoderCode = EncoderCode.GWN.value):
    encoderGenerator = STGNNEncoderGenerator(numberOfEpochs = epochs, env = env, savePath = savePath, encoderCode = encoderCode)
    encoderGenerator.generateModels()

# trains the DDQN model using whichever encoder was passed in.
def runDDQNAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph):
    global env
    modelPathName = savePath + 'DDQN'
    print("#"*27+"\n DDQN Trading Agent Training\n" + "#"*27)
    for seed in seeds:
        greedyPath = modelPathName + "_SEED "+str(seed)+"_greedy.pth"
        balancedPath = modelPathName + "_SEED "+str(seed)+"_balanced.pth"
        setSeed(seed)
        env.createLoader(shuffleTrain = False, batchSize = 32, formatStaticGraph = formatStaticGraph)
        ddqnAgent = DDQNTradingAgent(encoderModel = encoderModel, encoderCode = encoderCode, epochs = epochs, env = env)
        ddqnAgent.train(modelROI_Threshold = 0, modelSavePathGreedy = greedyPath, modelSavePathBalanced = balancedPath, env = env)
        torch.cuda.empty_cache()
        gc.collect()
        del ddqnAgent

# Executes the walk-forward testing on the DDQN model
def walkThroughDDQNAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph):
    global env
    modelPathName = savePath + 'DDQN'
    print("#"*27+"\n DDQN Trading Agent Walk-through\n" + "#"*27)
    for seed in seeds:
        modelPath = modelPathName + "_SEED "+str(seed)+"_balanced.pth"
        setSeed(seed)
        env.createLoader(shuffleTrain = False, batchSize = 32, formatStaticGraph = formatStaticGraph)
        ddqnAgent = DDQNTradingAgent(encoderModel = encoderModel, encoderCode = encoderCode, epochs = epochs, env = env)
        ddqnAgent.testWalkThroughTrain(modelSavePath = modelPath, env= env, walkEpochs = epochs)
        torch.cuda.empty_cache()
        gc.collect()
        del ddqnAgent

# trains the A2C model using whichever encoder was passed in.
def runA2CAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph):
    global env
    modelPathName = savePath + 'A2C'
    print("#"*27+"\n A2C Trading Agent Training\n" + "#"*27)
    for seed in seeds:
        greedyPath = modelPathName + "_SEED "+str(seed)+"_greedy.pth"
        greedyPathCritic = modelPathName + "_SEED "+str(seed)+"_greedyCritic.pth"
        balancedPath = modelPathName + "_SEED "+str(seed)+"_balanced.pth"
        balancedPathCritic = modelPathName + "_SEED "+str(seed)+"_balancedCritic.pth"
        setSeed(seed)
        env.createLoader(shuffleTrain = False, batchSize = 32, formatStaticGraph = formatStaticGraph)
        a2cAgent = A2CTradingAgent(epochs = epochs, encoderModel = encoderModel, encoderCode = encoderCode, env = env)
        a2cAgent.train(modelROI_Threshold = 0, modelSavePathGreedy = greedyPath, modelSavePathGreedyCritic = greedyPathCritic, modelSavePathBalanced =
                       balancedPath, modelSavePathBalancedCritic = balancedPathCritic, env = env)
        torch.cuda.empty_cache()
        gc.collect()
        del a2cAgent

# Executes the walk-forward testing on the A2C model
def walkThroughA2CAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph):
    global env
    modelPathName = savePath + 'A2C'
    print("#"*27+"\n A2C Trading Agent Walk-through\n" + "#"*27)
    for seed in seeds:
        modelActorPath = modelPathName + "_SEED "+str(seed)+"_balanced.pth"
        modelCriticPath = modelPathName + "_SEED "+str(seed)+"_balancedCritic.pth"
        setSeed(seed)
        env.createLoader(shuffleTrain = False, batchSize = 32, formatStaticGraph = formatStaticGraph)
        a2cAgent = A2CTradingAgent(epochs = epochs, encoderModel = encoderModel, encoderCode = encoderCode, env = env)
        a2cAgent.testWalkThroughTrain(modelActorPath = modelActorPath, modelCriticPath = modelCriticPath, env=env, walkEpochs = epochs)
        torch.cuda.empty_cache()
        gc.collect()
        del a2cAgent

# trains the PPO model using whichever encoder was passed in.
def runPPOAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph):
    global env
    modelPathName = savePath + 'PPO'
    print("#"*27+"\n PPO Trading Agent Training\n" + "#"*27)
    for seed in seeds:
        greedyPath = modelPathName + "_SEED "+str(seed)+"_greedy.pth"
        greedyPathCritic = modelPathName + "_SEED "+str(seed)+"_greedyCritic.pth"
        balancedPath = modelPathName + "_SEED "+str(seed)+"_balanced.pth"
        balancedPathCritic = modelPathName + "_SEED "+str(seed)+"_balancedCritic.pth"
        setSeed(seed)
        env.createLoader(shuffleTrain = False, batchSize = 32, formatStaticGraph = formatStaticGraph)
        ppoAgent = PPOTradingAgent(epochs = epochs, encoderModel = encoderModel, encoderCode = encoderCode, env = env)
        ppoAgent.train(modelROI_Threshold = 0, modelSavePathGreedy = greedyPath, modelSavePathGreedyCritic = greedyPathCritic, modelSavePathBalanced =
                       balancedPath, modelSavePathBalancedCritic = balancedPathCritic, env = env)
        torch.cuda.empty_cache()
        gc.collect()
        del ppoAgent

# Executes the walk-forward testing on the PPO model
def walkThroughPPOAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph):
    global env
    modelPathName = savePath + 'PPO'
    print("#"*27+"\n PPO Trading Agent Walk-through\n" + "#"*27)
    for seed in seeds:
        modelActorPath = modelPathName + "_SEED "+str(seed)+"_balanced.pth"
        modelCriticPath = modelPathName + "_SEED "+str(seed)+"_balancedCritic.pth"
        setSeed(seed)
        env.createLoader(shuffleTrain = False, batchSize = 32, formatStaticGraph = formatStaticGraph)
        ppoAgent = PPOTradingAgent(epochs = epochs, encoderModel = encoderModel, encoderCode = encoderCode, env = env)
        ppoAgent.testWalkThroughTrain(modelActorPath = modelActorPath, modelCriticPath = modelCriticPath, env=env, walkEpochs = epochs)
        torch.cuda.empty_cache()
        gc.collect()
        del ppoAgent

# trains all 3 RL agents using the specific encoder
def runRLAgents(savePath, encoderCode, epochs, encoderModel, formatStaticGraph):
    runDDQNAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph)
    runA2CAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph)
    runPPOAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph)

# Executes the walk-forward testing on all 3 RL agents using the passed in encoder
def runRLAgentsWalkThrough(savePath, encoderCode, epochs, encoderModel, formatStaticGraph):
    #walkThroughDDQNAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph)
    walkThroughA2CAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph)
    #walkThroughPPOAgent(savePath, encoderCode, epochs, encoderModel, formatStaticGraph)

# trains all models, across all encoders, all agents, all seeds, on the specified dataset
def trainModels(useJse = globalUseJSE):
    global env

    print("Training Models")
    
    if useJse:
        print("JSE")
        savePath = "data/modelData/JSE/"
    else:
        print("NYSE")
        savePath = "data/modelData/NYSE/"
    
    env = StockEnvironment(useJSE = useJse, flattenedNodes = True)
    genDNNEncoder(epochs = 200, env = env, savePath = savePath)
    loadDNNEncoderWeights(savePath)
    rlSavePath = savePath + "DNN_"
    print("\nTraining RL agents with DNN Encoder\n")
    runRLAgents(savePath = rlSavePath, encoderCode = EncoderCode.DNN.value, epochs = 750, encoderModel = baselineEncoderModel, formatStaticGraph = False)
    env = StockEnvironment(useJSE = useJse, flattenedNodes = False)
    genSTGNNEncoder(epochs = 400, env = env, savePath = savePath, encoderCode = EncoderCode.AGCRN.value)
    genSTGNNEncoder(epochs = 200, env = env, savePath = savePath)
    loadGWNEncoderWeights(savePath)
    rlSavePath = savePath + "GWN_"
    print("\nTraining RL agents with GWN Encoder\n")
    runRLAgents(savePath = rlSavePath, encoderCode = EncoderCode.GWN.value, epochs = 750, encoderModel = graphWavenetEncoderModel, formatStaticGraph = True)
    loadAGCRNEncoderWeights(savePath)
    rlSavePath = savePath + "AGCRN_"
    print("\nTraining RL agents with AGCRN Encoder\n")
    runRLAgents(savePath = rlSavePath, encoderCode = EncoderCode.AGCRN.value, epochs = 750, encoderModel = agcrnEncoderModel, formatStaticGraph = True)

# Executes walk-forwards testing for all models, across all encoders, all agents, all seeds, on the specified dataset
def testWalkThrough(useJse = globalUseJSE):
    global env

    print("Walk-Forward Testing")

    if useJse:        
        print("JSE")
        savePath = "data/modelData/JSE/"
    else:
        print("NYSE")
        savePath = "data/modelData/NYSE/"
        
    #env = StockEnvironment(useJSE = useJse, flattenedNodes = True)
    #loadDNNEncoderWeights(savePath)
    #rlSavePath = savePath + "DNN_"
    #print("\nWalk-through with DNN Encoder\n")
    #runRLAgentsWalkThrough(savePath = rlSavePath, encoderCode = EncoderCode.DNN.value, epochs = 12, encoderModel = baselineEncoderModel, formatStaticGraph = False)
    env = StockEnvironment(useJSE = useJse, flattenedNodes = False)
    #print("\nWalk-through with GWN Encoder\n")
    #loadGWNEncoderWeights(savePath)
    #rlSavePath = savePath + "GWN_"
    #runRLAgentsWalkThrough(savePath = rlSavePath, encoderCode = EncoderCode.GWN.value, epochs = 12, encoderModel = graphWavenetEncoderModel, formatStaticGraph = True)
    print("\nWalk-through with AGCRN Encoder\n")
    loadAGCRNEncoderWeights(savePath)
    rlSavePath = savePath + "AGCRN_"
    runRLAgentsWalkThrough(savePath = rlSavePath, encoderCode = EncoderCode.AGCRN.value, epochs = 12, encoderModel = agcrnEncoderModel, formatStaticGraph = True)
    
if __name__ == "__main__":
    #trainModels(globalUseJSE)
    testWalkThrough(globalUseJSE)