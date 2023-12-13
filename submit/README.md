# Samantha Brosler CS285 Final Project

## Instructions

### Generating Final Plots
- Run `plots.ipynb` to generate the final plots included in my final project paper. Make sure the data folder is on the path.

### Complete Analysis
- Run `analyses.ipynb` to execute all four analyses included in my final project. 
  - **Note:** `analyses.ipynb` sets `test_splits = 3` for computational speed. 
  - The final plots in the paper were averaged across at least 10 test splits.

## Info about .py Files
- `default_model.py`: Contains the default CNN-RNN model referenced in my final project paper.
- `evaluation.py`: Includes functions for evaluating the default and reward models.
- `plotting.py`: Contains functions for generating all of my plots.
- `preprocessing.py`: Includes functions for further processing the ECoG data.
- `training.py`: Contains functions for training models.

## Requesting Data

- The datasets `feedback_data.pkl` and `alphabet_with_fingerflex_data.pkl` are not included in the submission due to their large sizes (1.68 and 3.42 GB, respectively).
- To rerun all the analyses in `main.ipynb`, these datasets are necessary.
- However, they are not needed to regenerate the plots in `plots.ipynb`.
- For data requests, please contact me at [broslers@berkeley.edu](mailto:broslers@berkeley.edu).
