# TensorFlow Decision Forests inference on Arduino
 
<!--Tags: Arduino TensorFlowDecisionForests TensorFlow MachineLearning-->
 
This project shows how to train a Machine Learning model with TensorFlow Decision Forests (in python), and export the model to an Arduino to run predictions.
 
We train a model to score, between 0 and 10, the quality of red wine from its characteristics (e.g. pH, fixed volatile and citric acidities, sugar, sulphates, etc.). The model is trained from the wine experts annotations collected in the [Wine Quality's UCI ML repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). The model is a small regressive Gradient Boosted Trees with 20 trees. The top part of the first tree looks as follow:

<p align="center">
<img src="plot_first_tree.png" />
</p>
 
## How it works

The project works as follow:
 
1. Train and evaluate a model with TF-DF (in python).
1. Export the model into a compact binary format (converted written in python).
1. Store the model data in a `.h` file compatible with the Arduino compiler.
1. Compile and upload the model data and inference code on an Arduino.
1. Run the model on the Arduino.
 
The model is stored in the Arduino's Flash memory. During inference, each tree is copied to RAM, run and discarded. The final prediction is the sum of the tree individual predictions.
 
## Usage example
 
1. Install Python>=3.7 and TensorFlow Decision Forests `pip3 install tensorflow_decision_forests`.
1. Run `python3 train_model.py` to train and export the model. The model is exported to `run_model/exported_model.h`. If the file already exists, it is overridden.
1. Compile and upload `run_model/run_model.ino` to an Ardwino. An example is hard coded into `run_model.ino`. The model prediction is printed on the Serial monitor.
 
## Constraints
 
The code has a few constraints:
 
- Only support for Gradient Boosted Trees model with one dimensional output (e.g. binary classification, regression). No support for Random Forests models.
- Only output the un-linked model predictions. For example, returns the logit of a binary classification model (instead of a probability).
- Only support for numerical features. No support for categorical or categorical-set features.
- Limited to a maximum of 32k nodes per tree. Note: Each node takes 8 bytes of flash memory, so you'll probably run out of memory before that.
 
## Binary model format
 
The model binary format contains three parts:

1. The header (stored in Ram)
1. The node lists (stored in flash; one for each tree)
1. The address and size of each node lists 

### Header
 
| Size, bytes | Description |
|----|---|
| 2 | Format version |
| 2 | Number of trees, <num_trees> |
| 2 | Number of nodes, <num_nodes> |
 
### Node list address and size
 
| Size, bytes | Description |
|----|---|
| 2 x <num_trees> | Address of each tree's node list in flash memory |
 
| Size, bytes | Description |
|----|---|
| 2 x <num_trees> | Number of nodes in each tree |
 
### Node list (one for each tree)
 
| Size, bytes | Description |
|----|---|
| 8 x <num_nodes> | The node data |
 
Each node data is as follow:
 
| Size, bytes | Description |
|----|---|
| 2 | Index offset to the positive child node if the node is not a leaf. 0 if the node is a leaf. Note: The negative child node is always the next node. |
| 2 | Index of the input feature tested by the node if the node is not a leaf. |
| 4 | Float. Value of the node if the node is a leaf. Threshold to test if the node is not a leaf. |
 
Integer values are stored in little endian. This model format is inspired from [Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests/blob/main/yggdrasil_decision_forests/serving/decision_forest/decision_forest.h).