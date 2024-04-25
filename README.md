# Project Title

This project contains two main Python scripts: `FunctionsRevision.py` and `train_ml_estimators.py`.

## FunctionsRevision.py

This script contains various utility functions that are used throughout the project. These functions are designed to assist with data processing, model evaluation, and other tasks.

## train_ml_estimators.py

This script is used to train machine learning models. It uses the utility functions from `FunctionsRevision.py` to process the data and evaluate the models.

## Getting Started

To run these scripts, you will need Python 3.12.x installed on your machine. You will also need various Python libraries, which can be installed with conda:

```bash
conda env create -f environment.yml
```
Double check thst the `name` and `prefix` fields comply with your desired environment name and path.

Then, switch to the environment:
```bash
conda activate <env_name>
```

## Running the Scripts

To run the scripts, navigate to the directory containing the scripts and use the python command:

```bash
python train_ml_estimators.py
python FunctionsRevision.py
```

## ML-framework configuration

Check the flags of the `train_ml_estimators.py` to especify the model targets and the ML algorithms to be tested. Default values:

```python
# Default values
config_file = './config'
dataset_file = r'./data/datasetProm.json'
feature_modes = ['FS', 'FE', 'NoFE']
algorithms = ['RF', 'SVR', 'RR', 'NN', 'ABR', 'KNR', 'SVC', 'GNB']
targets = ['rtt', 'videoWidth', 'initPlayingTime', 'videoDisplayRate', 'avgStallTimeFixed', 'throughput', 'bufferDuration']
save_model = True
use_kqi = False
scale = True
save_dataset = True
```

Also, if you want to add new ML algorithms, you probably should edit some functions (adding hyperparameters to be evaluated) 

```python
# Create a model object with an parameter grid for GridSearchCV tuning
if algorithm == 'RF':
        # Create a model object
        model = RandomForestRegressor(random_state = 0)
        # Define the parameter space that will be searched over
        param_grid = {
                        'model__n_estimators' : np.arange(10, 101, 10),
                        'model__max_depth' : np.arange(5, 10)
                                }
```

Finally, check the `config.json` has configured the all the targets you want to be split from the input features and the features you want to be discarded (i.e., textual info, useless data, etc.). 
An example for 360-Video can be seen below:

```python
{
    "KQI_types": {
        "initPlayingTime": "continuous",
        "videoWidth": "discrete",
        "resolutionSwitches": "continuous",
        "res5": "continuous",
        "res4": "continuous",
        "res3": "continuous",
        "res0": "continuous",
        "res1": "continuous",
        "res2": "continuous",
        "resolution": "discrete",
        "resProfile": "continuous",
        "videoDisplayRate": "continuous",
        "estimatedTotalBandwidthUsed": "continuous",
        "stallCount": "continuous",
        "stallEvents": "continuous",
        "stallTime": "continuous",
        "throughput": "continuous",
        "bufferDuration": "continuous",
        "rtt": "continuous",
        "avgStallTimeFixed": "continuous"
    },
    "Drop_features": [
        "time",
        "xGlobal",
        "time",
        "UE_UE_ul_retx",
        "UE_UE_dl_retx",
        "ue_count_max",
        "ue_count_min",
        "freqs",
        "total"
    ]
}
```

## Contributing

Contributions are welcome. Please open an issue to discuss your idea or submit a pull request.

## License

This project is licensed under the terms of the MIT license.
