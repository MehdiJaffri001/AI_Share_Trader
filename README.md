The code attached is for the AI-Share trader Honours Project by JFFSYE002.

The file structure format is as follows:

├── data/                  # Stores project data

│   ├── modelData/         # Stores models produced from training and walk-forward testing (model not included in submission to save space.)

│   └── rawData/           # Stores excel with raw data. (Excel raw data not included in submission due to legal sharing restrictions.) 

|

├── src/                   # Stores source files used in the project                   

│   ├── Environment/       # Environment that processes raw data and prepares it for encoders.

│   ├── Encoders/          # Contains scripts that generate the encoder models and train them. 

│   └── RLAgents/          # Contains scripts for Rl agnets to use the encoders to train and walk-forward respectively.

|

├── main.py                # Master script that manages creation and running of all models. 

├── projectRunning.ipynb   # Jupyter notebook used to run the main.py file and generate plots from output. 

├── requirements.txt       # Core library dependencies

└── README.md              # This file which you are currently reading. 


*Note: modelData has 2 subfolders, one for each dataset.

       RLAgents have further classes and splits such as continuous and discrete agents before the final child classes which are the respective algorithms.
       
       Training time are long even with a GPU. 

############################################

# AI- Share Trader Honours Project

############################################

Setup:

We have 2 datasets: JSE and NYSE

We have 3 encoders: DNN, GWN and AGCRN

We have 4 agents: Rule-Based, DDQN, A2C and PPO


We simulate each model on each dataset, train them and then update them using walk-forward testing.
To run it yourself, you need to just run the notebook cell with '%run main.py'.
It defualts to JSE but you can specify it like so '%run main.py JSE'.
Alternatively use NYSE but which is specified like so '%run main.py NYSE'.
The cell in the notebook is configured to run both but the notebook is designed to be able to be changed so that we can choose what to run freely.

Core Results:

As an overall summary we found that while NYSE and JSE were profitable, JSE's greater volatility provided a chance for returns that would not be possible on the NYSE. The model that performed the best in terms of all metrics on the JSE was the AGCRN-PPO model which beat the Top40 Index in terms of Raw ROI, Sortino score and Calmar scores. This was also the only model that beat the Index in all 3 metrics on the JSE.  
