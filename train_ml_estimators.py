# Import the libraries to supress warnings
import warnings
warnings.simplefilter("ignore")

# Importing the libraries
from FunctionsRevision import Functions as fun
import argparse

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


# Main function
def main(dataset_file , config_file, save_model, use_kqi, scale, save_dataset, feature_modes, algorithms, targets):
    
    
    # Read the configuration file
    configuration = fun.load_configuration(config_file)

    # Configure the experiment
    KQIs, drop_features, feature_modes, algorithms, targets, KQI_type = fun.configure_ml_experiment(configuration_dict=configuration, feature_modes=feature_modes, algorithms=algorithms, targets=targets)

    # Load the dataset
    dataset, Y = fun.load_dataset(dataset_file, targets=KQIs, drop_features=drop_features)

    # Create FS/FE datasets
    X_train_FE, X_test_FE, X_train_FS, X_test_FS, y_train, y_test = fun.create_FE_FS_dataset(X=dataset, Y=Y, feature_modes=feature_modes, scale=scale, save_dataset=save_dataset)

    # Train the models
    fun.train_models(X_train_FE=X_train_FE, X_train_FS=X_train_FS, X_test_FE=X_test_FE, X_test_FS=X_test_FS, y_train=y_train, y_test=y_test,
                    algorithms=algorithms, feature_modes=feature_modes, KQI_type=KQI_type, targets=targets, save_model=save_model, use_kqi=use_kqi)
  

# Call main function
if __name__ == "__main__":

    # Create a parser for arguments and options
    parser = argparse.ArgumentParser(description='Arguments for the train_ml_estimators.py script. This script trains the combination of feature selection and feature extraction methods with the machine learning algorithms.' + 
                                     'The configuration of the targets and the algorithms is done in the configuration file. The configuration MUST be updated if the dataset changes. The configuration file is a JSON file' + 
                                     'The configuration file MUST be in the same directory as the script. The configuration file MUST be named as config.json.')
    
    # Add the arguments
    parser.add_argument('--config', type=str, help='Path to the configuration file, by default the root directory.', default=config_file)
    parser.add_argument('--save_model', type=bool, help='Save the trained models, by default True.', default=save_model)
    parser.add_argument('--use_kqi', type=bool, help='Use the KQI as target, by default False. CURRENTLY NOT IMPLEMENTED', default=use_kqi)
    parser.add_argument('--scale', type=bool, help='Scale the dataset, by default True.', default=scale)
    parser.add_argument('--save_dataset', type=bool, help='Save the datasets, by default True.', default=save_dataset)
    parser.add_argument('--feature_modes', type=list, help='Feature selection and feature extraction methods. Options: FE, FS or NoFE. By default: [FE and FS]', default=feature_modes)
    parser.add_argument('--algorithms', type=list, help='Machine learning algorithms, by default [KNR, RF]. Options available: RF, SVR, NN, ABR, RR, KNR, SVC, GNB', default=algorithms)
    parser.add_argument('--targets', type=list, help='Targets to train the models, by default [videoWidth, rtt].', default=targets)
    parser.add_argument('--dataset_file', type=str, help='Path to the dataset file, by default in the root/data directory.', default=dataset_file)

    # Parse the command line arguments
    args = parser.parse_args()

    # Assign the values to the variables
    config_file = args.config
    save_model = args.save_model
    use_kqi = args.use_kqi
    scale = args.scale
    save_dataset = args.save_dataset
    feature_modes = fun.map_acronym_to_feature_eng_mode(args.feature_modes)
    algorithms = args.algorithms
    targets = args.targets
    dataset_file = args.dataset_file

    # Call the main function
    main(dataset_file=dataset_file, config_file=config_file, save_model=save_model, use_kqi=use_kqi, scale=scale, save_dataset=save_dataset, feature_modes=feature_modes, algorithms=algorithms, targets=targets)