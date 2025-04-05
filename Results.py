import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Define models
def build_model_one_hidden():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_three_hidden():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Helper to train and evaluate
def train_and_evaluate(model, x_train, y_train, x_test, y_test, y_test_raw, label):
    print(f"\nTraining: {label}")
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2)
    train_time = time.time() - start_time

    # Predictions
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    cm = confusion_matrix(y_test_raw, y_pred)
    report = classification_report(y_test_raw, y_pred, output_dict=True)
    avg_precision = report['weighted avg']['precision']
    avg_recall = report['weighted avg']['recall']
    avg_f1 = report['weighted avg']['f1-score']
    final_acc = history.history['val_accuracy'][-1]
    final_loss = history.history['val_loss'][-1]
    total_params = model.count_params()

    print(classification_report(y_test_raw, y_pred))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {label}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return {
        'Model': label,
        'Final Accuracy': final_acc,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F1 Score': avg_f1,
        'Train Time (s)': round(train_time, 2),
        'Loss': final_loss,
        'Params': total_params,
        'History': history
    }

# Train on full dataset
results = []
results.append(train_and_evaluate(build_model_one_hidden(), x_train, y_train_cat, x_test, y_test_cat, y_test, '1 Layer (Full)'))
results.append(train_and_evaluate(build_model_three_hidden(), x_train, y_train_cat, x_test, y_test_cat, y_test, '3 Layers (Full)'))

# Train on reduced dataset (50%)
x_half = x_train[:30000]
y_half = y_train_cat[:30000]
results.append(train_and_evaluate(build_model_one_hidden(), x_half, y_half, x_test, y_test_cat, y_test, '1 Layer (Half)'))
results.append(train_and_evaluate(build_model_three_hidden(), x_half, y_half, x_test, y_test_cat, y_test, '3 Layers (Half)'))

# Plot accuracy comparisons
plt.figure()
for r in results:
    plt.plot(r['History'].history['val_accuracy'], label=r['Model'])
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Results DataFrame
df_results = pd.DataFrame(results).drop(columns=['History'])
print("\n=== Summary Results ===")
print(df_results)
