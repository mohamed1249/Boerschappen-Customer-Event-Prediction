# Boerschappen Customer Event Prediction

This repository contains code and resources for predicting customer events using an LSTM model. The dataset consists of customer events from Boerschappen, and the project involves data preparation, feature engineering, model training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict customer events for Boerschappen using a machine learning model. The dataset includes various customer events, and the goal is to build a predictive model to forecast future events based on historical data.

## Data Preparation

The data preparation process involves several steps, including data cleaning, feature engineering, and visualization. Below is a brief overview of the steps:

1. **Loading Data**: The dataset is loaded using pandas.
2. **Initial Exploration**: Basic exploration of the dataset to understand its structure and contents.
3. **Data Cleaning**: Handling missing values and ensuring data types are consistent.
4. **Feature Engineering**: Creating new features, such as time since last event and encoding categorical variables.
5. **Visualization**: Visualizing distributions and relationships within the data using matplotlib, seaborn, and plotly.

### Example Code

```python
import pandas as pd

# Load dataset
df = pd.read_csv('Boerschappen-Newdata-Box-Met-Events.csv')

# Basic info
df.info()

# Select relevant columns
df = df.iloc[:,:11]

# Convert timestamps to datetime
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df['event_date'] = pd.to_datetime(df['event_timestamp'].dt.date)

# Sort data
df = df.sort_values(['customer_id', 'event_timestamp'])

# Calculate time since last event
df['last_event'] = df.groupby('customer_id')['event'].shift(1)
df['last_event_date'] = pd.to_datetime(df.groupby('customer_id')['event_date'].shift(1))
df['time_since_last_event'] = (df['event_date'] - df['last_event_date']).dt.days

# Visualize event counts
df['event'].value_counts().plot(kind='bar', figsize=(20,10))

# Drop missing values
df.dropna(inplace=True)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
cat_cols = [cat for cat in df.columns if df[cat].dtype == 'object' and cat != 'customer_id']
df_ = df.copy()
for cat in cat_cols:
    label_encoder.fit(df[cat].astype(str))
    df_[cat] = label_encoder.transform(df_[cat].astype(str))

# Save prepared data
df.to_csv('preparedData.csv', index=False)
```

## Modeling

The modeling process involves building and training an LSTM model using TensorFlow. The model is designed to predict customer events based on the prepared dataset.

### Example Code

```python
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Load prepared data
data = pd.read_csv('ready_to_model_df.csv')

# Prepare training data
X = data.drop(['event', 'time_since_last_event', 'customer_id'], axis=1)
y = data[['event', 'time_since_last_event']]
y['combined'] = y.event.astype(str) + '-' + y.time_since_last_event.astype(str)

# Encode target variable
label_encoder = LabelEncoder()
y['encoded'] = label_encoder.fit_transform(y['combined'])

# Resample to balance classes
data_ = np.column_stack((X, y['encoded']))
minority_class = y['encoded'].value_counts().median() * 2
resampled_data = [resample(data_[data_[:, -1].astype(int) == i], n_samples=int(minority_class)) for i in y.encoded.unique()]
balanced_data = np.vstack(resampled_data)

# Split data into training and validation sets
X_balanced = balanced_data[:, :-1]
y_balanced = balanced_data[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Build LSTM model
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(256),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(88, activation='softmax')
    ])
    return model

# Train LSTM model
input_shape = (X_train.shape[1], 1)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)).astype(np.float32)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=88)

model = build_lstm_model(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Save model
model.save('LSTM_model.h5')
```

## Results

The results of the model training and evaluation include accuracy scores (86%, 63%), confusion matrices, and visualizations of predicted versus actual events. Key metrics and visualizations are provided to assess the model's performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
