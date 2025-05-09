# 1. Generate Dataset

OpenVLA use RLDS format to load data. Need to manually process current data into their supported format.

First change the data loading path in `rlbench_data_util/rlbench_dataset_builder.py` line 77-78, and the save dir in `visualize_dataset.py` line 3. Then run
```
python visualize_dataset.py
```


# Note: About data format

In openvla's pipeline, the data format is different from the my training script in llama-factory. In general, it uses euler angle to represent rotation and use delta values instead of absolute values.

Some places that need to pay attention to:

1, `rlbench_data_util/rlbench_dataset_builder.py`, line 122. Here we load raw data into rlds format.

2, `prismatic/vla/datasets/rlds/dataset.py`, line 39. Here we load the data from previous stage into for training, during which we call `prismatic.vla.datasets.rlds.oxe.transforms.rlbencho1_dataset_transform` in line 138 to transform quaternions to euler angles and calculate delta.

3, `prismatic/vla/datasets/datasets.py`, line 43. Process the data from previous step into vla's input, such as action tokenization and prompt formatting.



# 2. Train Model
```
bash mytrain.sh
```