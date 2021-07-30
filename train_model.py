# Train a Gradient Boosted Tree model with the TensorFlow Decision Forests
# library. Then, export the model to an Ardwino compatible serving format.
#
# Mathieu Guillame-Bert, 2021

import tensorflow_decision_forests as tfdf
import pandas as pd
import tensorflow as tf
from lib import atfdf

print("TensorFlow Decision Forests v" + tfdf.__version__)

dataset_path = tf.keras.utils.get_file(
    "wine.csv", "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")

# Load the training dataset.
train_df = pd.read_csv(dataset_path, sep=";")

print("The first three training examples:")
print(train_df.head(3))

# Converts the Pandas Dataframe into a Tensorflow Dataset.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    train_df, label="quality", task=tfdf.keras.Task.REGRESSION)

# Train the model.
model = tfdf.keras.GradientBoostedTreesModel(
    num_trees=20,
    max_num_nodes=32,
    task=tfdf.keras.Task.REGRESSION,
    growing_strategy="BEST_FIRST_GLOBAL",
    shrinkage=0.2)
model.fit(train_ds)

print("Model statistics")
print(model.summary())

# Plot the top of the first tree.
html = tfdf.model_plotter.plot_model(model, tree_idx=0, max_depth=3)
open("plot_first_tree.html", "w").write(html)

# Note: By default, part of the training dataset is used for validation.
# In small datasets, this logic should be optimizer (and possibly disabled) to
# maximize the model quality.
print("Expected RMSE of the model (the lower the better):")
print(model.make_inspector().evaluation().rmse)

# Export the model.
#
# Training larger (number of trees and maximum depth) forests leads to better
# predictions. However, the model size is limited by the Ardwino memory and
# increase the inference speed. The size of the model is printed bellow. It is
# also available in the .h generated file.
atfdf.convert_model(model, "run_model/exported_model.h")

# Show the first example values and prediction.
# The Ardwino "predict" method will return the same value.
print("Attributes and prediction of the first example:")
for example, _ in train_ds.unbatch().batch(1).take(1):
    print("Attributes:")
    for name, value in example.items():
        print(f"\t{name}: {value[0]}")
    print("Prediction: ", model.predict(example))
