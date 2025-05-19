# ANLP Project - Real Estate Price Prediction Using Large Language models
This project aims to predict property prices using the properties' textual overview for enhanced accuracy

**In order to run our files, please create a new pip/conda envinronment with python 3.9 and install the required packages using `pip install -r requirements.txt`**

## Runnable files
- **src/zero_shot.py**: Runs a zero shot evaluation on pre trained models and saves the results to a file
- **src/fine_tuning.py**: Runs a fine tuning process with our training data. print the evaluation results thoughtout the process but doesn't save anything
- **src/evaluate_models.py**: Fine tunes each of the pre-defined models and evaluates the fine-tuned models on out test data. The statistics are saved to a results file at the end of the process.
- **src/preprocessing.py**: Creates the train, validation and test datasets from the raw data. If you are cloning this repository, the files which are created from this script should already be present (which are `train_data.csv`, `validation_data.csv`, `test_data_in_dist.csv`, `test_data_out_dist.csv`), so running this script again shouldn't do anything special
- **src/visualizaion.py**: Outputs graphs analyzing our data

Collaborators: Avishai Elmakies, Elad Malik, Eden Kalij & Noam Ben Sason.

