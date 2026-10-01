import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -------------------------
# SETTINGS
# -------------------------

DATASET_PATH = "dataset"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10


# -------------------------
# LOAD DATASET
# -------------------------

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

print()
print("Classes found:")
print(class_names)
print()


# -------------------------
# CREATE CNN MODEL
# -------------------------

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),

    # Change pixel values from 0-255 to 0-1
    layers.Rescaling(1.0 / 255),

    # First convolution layer
    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Second convolution layer
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convert image features into numbers
    layers.Flatten(),

    # Hidden layer
    layers.Dense(64, activation="relu"),

    # Output layer
    layers.Dense(len(class_names), activation="softmax")
])


# -------------------------
# COMPILE MODEL
# -------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -------------------------
# TRAIN MODEL
# -------------------------

print()
print("Starting training...")
print()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# -------------------------
# CHECK ACCURACY
# -------------------------

loss, accuracy = model.evaluate(val_ds)

print()
print("Validation accuracy:", accuracy)
print("Validation accuracy percentage:", accuracy * 100, "%")
print("Validation loss:", loss)


# -------------------------
# SAVE MODEL
# -------------------------

model.save("circuit_model.keras")

print()
print("Model saved as circuit_model.keras")


# -------------------------
# GET PREDICTIONS
# -------------------------

y_true = []
y_pred = []

wrong_images = []
wrong_true = []
wrong_pred = []

print()
print("Generating predictions...")

for images, labels in val_ds:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    for i in range(len(labels)):

        true_label = int(labels[i].numpy())
        predicted_label = int(predicted_classes[i])

        y_true.append(true_label)
        y_pred.append(predicted_label)

        # Save incorrect predictions
        if true_label != predicted_label:

            wrong_images.append(
                images[i].numpy()
            )

            wrong_true.append(
                true_label
            )

            wrong_pred.append(
                predicted_label
            )


# -------------------------
# CONFUSION MATRIX
# -------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues"
)

plt.title("Circuit Component Confusion Matrix")

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()

print()
print("Saved confusion_matrix.png")


# -------------------------
# PRINT CONFUSION MATRIX
# -------------------------

print()
print("Confusion Matrix:")
print(cm)


# -------------------------
# FAILURE ANALYSIS
# -------------------------

print()
print(
    "Total wrong predictions:",
    len(wrong_images)
)


# Save first 5 mistakes

number_to_save = min(
    5,
    len(wrong_images)
)

for i in range(number_to_save):

    plt.figure(figsize=(5, 5))

    plt.imshow(
        wrong_images[i].astype("uint8")
    )

    true_name = class_names[
        wrong_true[i]
    ]

    predicted_name = class_names[
        wrong_pred[i]
    ]

    plt.title(
        "True: "
        + true_name
        + "\nPredicted: "
        + predicted_name
    )

    plt.axis("off")

    plt.tight_layout()

    filename = (
        "failure_"
        + str(i + 1)
        + ".png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

    print(
        "Saved:",
        filename
    )


# -------------------------
# COUNT TYPES OF MISTAKES
# -------------------------

print()
print("Failure Summary:")

failure_counts = {}

for true_label, predicted_label in zip(
    wrong_true,
    wrong_pred
):

    true_name = class_names[
        true_label
    ]

    predicted_name = class_names[
        predicted_label
    ]

    mistake = (
        true_name
        + " -> "
        + predicted_name
    )

    if mistake in failure_counts:
        failure_counts[mistake] += 1

    else:
        failure_counts[mistake] = 1


for mistake, count in sorted(
    failure_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        mistake,
        ":",
        count
    )


# -------------------------
# TRAINING ACCURACY GRAPH
# -------------------------

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "accuracy_graph.png",
    dpi=300
)

plt.close()

print()
print("Saved accuracy_graph.png")


# -------------------------
# TRAINING LOSS GRAPH
# -------------------------

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "loss_graph.png",
    dpi=300
)

plt.close()

print()
print("Saved loss_graph.png")


# -------------------------
# FINISHED
# -------------------------

print()
print("Finished!")
print()

print("Generated files:")

print(
    "- circuit_model.keras"
)

print(
    "- confusion_matrix.png"
)

print(
    "- accuracy_graph.png"
)

print(
    "- loss_graph.png"
)

for i in range(number_to_save):

    print(
        "- failure_"
        + str(i + 1)
        + ".png"
    )
