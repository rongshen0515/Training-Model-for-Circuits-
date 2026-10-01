import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_PATH = "dataset"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

class_names = train_ds.class_names

print("Classes found:")
print(class_names)

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),

    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

loss, accuracy = model.evaluate(val_ds)

print("Validation accuracy:", accuracy)

model.save("circuit_model.keras")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------
# CONFUSION MATRIX
# -------------------------

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -------------------------
# SAVE 5 WRONG PREDICTIONS
# -------------------------

wrong_count = 0

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    for i in range(len(labels)):
        true_label = labels[i].numpy()
        predicted_label = predicted_classes[i]

        if true_label != predicted_label:

            plt.figure()

            plt.imshow(images[i].numpy().astype("uint8"))

            plt.title(
                "True: "
                + class_names[true_label]
                + " | Predicted: "
                + class_names[predicted_label]
            )

            plt.axis("off")
            plt.tight_layout()

            plt.savefig("failure_" + str(wrong_count + 1) + ".png")

            wrong_count += 1

            if wrong_count == 5:
                break

    if wrong_count == 5:
        break

print("Saved confusion matrix and failure examples.")