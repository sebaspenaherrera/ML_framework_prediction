# Project Title

This project contains three main components: `train_ml_estimators.py`, `DrawMLResults.ipynb`, and `FunctionsRevision.py`.

## train_ml_estimators.py

This script is used to train machine learning models. It uses various utility functions to process the data, train the models, and evaluate their performance.

## DrawMLResults.ipynb

This Jupyter notebook is used to visualize the results of the machine learning models trained by `train_ml_estimators.py`. It generates various plots and metrics to help you understand the performance of the models.

```python
# Get the feature scores
feature_scores = (fun.get_feature_scores(targets=targets)).reset_index(inplace=False, drop=True)

# Calculate the MI between input features and targets and input features
MI = fun.get_mi_inputs_targets_baseline(targets=targets)

# Plot the MI matrix
fun.draw_heat_map(MI_df = MI, title='Mutual Information Matrix', cbar_label='Information (nat)', path='./figures/2024-04-02/', height_cm=20)
```

## FunctionsRevision.py

This script contains various utility functions that are used throughout the project. These functions assist with data processing, model training, evaluation, and visualization tasks.

## Getting Started

To run these scripts, you will need to have Anaconda installed on your machine and a Conda environment set up.

1. Install the required packages in your Conda environment:

```bash
conda create --name <env> --file requirements.txt
```

2. Activate the Conda environment:

```bash
conda activate <env>
```

3. Run the training script:

```bash
python train_ml_estimators.py
```

4. Open the Jupyter notebook to visualize the results:

```bash
jupyter notebook DrawMLResults.ipynb
```

## Contributing

Contributions are welcome. Please open an issue to discuss your idea or submit a pull request.

## License

This project is licensed under the terms of the MIT license.