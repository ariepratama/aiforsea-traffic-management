# Ai For SEA - Traffic Management Challenge

This is my solution to aiforsea.com traffic management challenge.

# Quickstart - How To Run Final Predictions

1. Run `pip install -r requirements.txt`
2. Run `python trainer.py [path_to_train_data]`
3. Result will be stored by default at `result.csv` at the same folder as current working directory

## Modifying `trainer.py` parameters

It is possible to modify these parameters when running `trainer.py`

|Parameter Name|Default Value|Description|
|---|---|---|
|--prediction-future-ticks|5|1 ticks is 15 minutes for all geo|
|--training-start-from-day|1|start training from which day|
|--training-day-length|14|how many days should the training consists of|
|--prediction-result-path|result.csv|where to store prediction result dataframe|


# Analysis
see at [ANALYSIS.md](ANALYSIS.md)
