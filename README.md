# Samantha Brosler CS285 Final Project

## Instructions

### Generating Final Plots
- `plots.ipynb` generates the final plots included in my final project paper from saved data in the 'data' folder

### Complete Analysis
- `main.ipynb` executes all four analyses discussed in my final project. 
  - **Note:** `main.ipynb` sets `test_splits = 3` for computational speed. 
  - The final plots in the paper were averaged across at least 10 test splits.

## Info about .py Files
- `default_model.py`: Contains the default CNN-RNN model referenced in my final project paper.
- `evaluation.py`: Includes functions for evaluating the default and reward models.
- `plotting.py`: Contains functions for generating all of my plots.
- `preprocessing.py`: Includes functions for further processing the ECoG data.
- `training.py`: Contains functions for training models.

## Requesting data
- I did not include the feedback_data.pkl and alphabet_with_fingerflex_data.pkl datasets in my submission because they are very large (1.68 and 3.42 GB respectively)
  - feedback_data.pkl is necessary for the Analysis 4 figures in `plots.ipynb`
  - alphabet_with_fingerflex_data.pkl is necessary to rerun `main.ipynb
- Please contact me for data requests (broslers@berkeley.edu)
