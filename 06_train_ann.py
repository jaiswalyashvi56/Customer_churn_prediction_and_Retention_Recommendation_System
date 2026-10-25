

import pandas as pd
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Load cleaned data
df = pd.read_csv("data/cleaned_churn.csv")

# Step 2: Prepare input features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Step 3: Encode categorical features
X = pd.get_dummies(X, dtype=int)

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 5: Scale the data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Build the Artificial Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Step 7: Configure model training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Step 8: Stop training when validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Step 9: Give more weight to churn customers
class_weights = {
    0: 1.0,
    1: 2.0
}

# Step 10: Train the model
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weights,
    verbose=1
)

# Step 11: Evaluate on unseen test data
test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\n--- Model Evaluation ---")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))

# Step 12: Generate predictions
probabilities = model.predict(X_test_scaled, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype(int)

print("\n--- Classification Report ---")
print(classification_report(
    y_test,
    predictions,
    target_names=["No Churn", "Churn"]
))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))

# Step 13: Save files for the Streamlit application
model.save("churn_ann.keras")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\nANN model and preprocessing files saved successfully!")
