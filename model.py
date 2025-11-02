import numpy as np 
import pandas as pd 
from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
df = pd.read_csv("./dataset.csv")
df.head()

df.drop(['Fan_actuator_OFF','Fan_actuator_ON','Water_pump_actuator_OFF','Water_pump_actuator_ON'],axis=1,inplace=True)

df.head()

df.dropna(subset=['date'], inplace=True)

df.isna().sum()

df['date'] = pd.to_datetime(df['date'])

df.info()

df.set_index('date', inplace=True)

df.info()

df = df.sort_index()
df = df[~df.index.duplicated(keep='first')]

intervals = df.index.to_series().diff()


print(intervals)

print(intervals.value_counts())

df['pump_status'] = df['Watering_plant_pump_ON']


df = df.drop(columns=['Watering_plant_pump_ON', 'Watering_plant_pump_OFF'])

df

df.isna().sum()

df



X = df.drop(columns=['pump_status']).values
y = df['pump_status'].values


split_idx = int(0.8 * len(df))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

df = df.dropna()

df.isna().sum()

df['pump_status'].value_counts()

df.isna().sum()



# Initialize the model
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Prepare dictionary to store evaluation results
eval_results = {}

# Fit model with evaluation sets
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# ✅ Access eval results from the trained model directly
eval_results = model.evals_result()

# Extract losses
train_loss = eval_results['validation_0']['logloss']
test_loss  = eval_results['validation_1']['logloss']

# Plot
plt.figure(figsize=(8,5))
plt.plot(train_loss, label='Train Loss', linewidth=2)
plt.plot(test_loss, label='Test Loss', linewidth=2)
plt.title('XGBoost Training vs Testing Loss')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.show()



# Predict probabilities and classes
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)



last_input = df.drop(columns=['pump_status']).values[-1].reshape(1, -1)
next_prob = model.predict_proba(last_input)[:, 1][0]
next_class = int(next_prob > 0.5)
print(f"Predicted probability for next time step: {next_prob:.4f}")
print(f"Predicted class for next time step: {next_class}")


# Save the trained model
joblib.dump(model, "xgb_pump_model.pkl")
print("Model saved to xgb_pump_model.pkl")