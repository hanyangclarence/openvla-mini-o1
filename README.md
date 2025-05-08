# 1. Generate Dataset

OpenVLA use RLDS format to load data. Need to manually process current data into their supported format.

First change the data loading path in `rlbench_data_util/rlbench_dataset_builder.py` line 77-78, and the save dir in `visualize_dataset.py` line 3. Then run
```
python visualize_dataset.py
```


# 2. Train Model
```
bash mytrain.sh
```