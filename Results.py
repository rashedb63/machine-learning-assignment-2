# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical

# Step 1: Load and Preprocess Data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
           'BMI','DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv(url, names=columns)

features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
            'BMI','DiabetesPedigreeFunction','Age']
X = df[features].values
y = df['Outcome'].values
y_cat = to_categorical(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Step 2: Define Hyperparameter Grid
# Change the number of hidden layers and specify the range of parameters if necessary
param_grid = {
    'hidden_layers': [[128, 64, 32]],
    'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005],
    'momentum': [0.0, 0.5, 0.7, 0.9, 0.95],
    'batch_size': [32, 64],
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'dropout_rate': [0.0, 0.2]
}

# Step 3: Optimizer Factory
def get_optimizer(name, lr, momentum):
    if name == 'adam':
        return Adam(learning_rate=lr)
    elif name == 'sgd':
        return SGD(learning_rate=lr, momentum=momentum)
    elif name == 'rmsprop':
        return RMSprop(learning_rate=lr)
    else:
        raise ValueError("Unsupported optimizer")

# Step 4: Build Model
def build_model(input_shape, hidden_layers, dropout_rate):
    model = Sequential()
    model.add(Input(shape=input_shape))
    for units in hidden_layers:
        model.add(Dense(units, activation='relu'))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='softmax'))
    return model

# Step 5: Run Sensitivity Analysis
results = []
combinations = list(itertools.product(
    param_grid['hidden_layers'],
    param_grid['learning_rate'],
    param_grid['momentum'],
    param_grid['batch_size'],
    param_grid['optimizer'],
    param_grid['dropout_rate']
))

for idx, (hlayers, lr, mom, bsize, opt, drop) in enumerate(combinations):
    try:
        print(f"Training config {idx+1}/{len(combinations)}: Layers={hlayers}, LR={lr}, MOM={mom}, OPT={opt}, BS={bsize}, DO={drop}")
        model = build_model((8,), hlayers, drop)
        optimizer = get_optimizer(opt, lr, mom)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        start = time.time()
        history = model.fit(x_train, y_train, epochs=20, batch_size=bsize, verbose=0, validation_data=(x_test, y_test))
        end = time.time()

        y_pred = np.argmax(model.predict(x_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        val_loss = history.history['val_loss'][-1]
        train_time = round(end - start, 2)

        results.append({
            'Hidden Layers': str(hlayers),
            'Learning Rate': lr,
            'Momentum': mom,
            'Batch Size': bsize,
            'Optimizer': opt,
            'Dropout': drop,
            'Accuracy': acc,
            'F1 Score': f1,
            'Val Loss': val_loss,
            'Train Time (s)': train_time
        })
    except Exception as e:
        print(f"Error on config {idx+1}: {e}")

# Step 6: Display Results Table
df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values(by='Accuracy', ascending=False)
display(df_results_sorted.head(10))  # Show top 10

# Step 7: Plot Accuracy of Top Configs
plt.figure(figsize=(12, 6))
sns.barplot(data=df_results_sorted.head(10), x='Accuracy', y='Hidden Layers', hue='Optimizer')
plt.title('Top 10 Configurations by Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Hidden Layer Configuration')
plt.legend(title='Optimizer')
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Save results to CSV
#df_results.to_csv("diabetes_parameter_sensitivity_results.csv", index=False)
