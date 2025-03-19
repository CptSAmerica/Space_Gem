from space_gem.ml_logic.modeling import *
import os
from tensorflow.keras.utils import image_dataset_from_directory


batch_size = 64

# Creating training and val data sets
data_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
train_data_dir = os.path.join(data_path, "raw_data", "train")
test_data_dir = os.path.join(data_path, "raw_data", "test")

train_ds = image_dataset_from_directory(
    train_data_dir,
    labels="inferred",
    label_mode="categorical",
    seed=123,
    image_size=(225, 225),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    test_data_dir,
    labels="inferred",
    label_mode="categorical",
    seed=123,
    image_size=(225, 225),
    batch_size=batch_size
)

# Fitting, and testing the baseline model

baseline_model = initialize_baseline_model()
baseline_model = compile_model(baseline_model)
baseline_model, baseline_history = train_model(
        model=baseline_model,
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=batch_size,
        patience=10
    )

# Fitting, and testing the current model

model = initialize_model()
model = compile_model(model)
model, history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=batch_size,
        patience=10
    )

print(f"Baseline model's val_accuracy is:  {round(np.min(baseline_history.history['val_accuracy']), 2)}")
print(f"Currnet model's val_accuracy is:  {round(np.min(history.history['val_accuracy']), 2)}")
