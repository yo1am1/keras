import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

data = pd.read_csv("kc_house_data.csv")

# Видалення непотрібних стовпців
data = data.drop(["date"], axis=1)

data_normalised = preprocessing.normalize(data, axis=0)
data_normalised = pd.DataFrame(data_normalised, columns=data.columns)

# Ознаки та цільова змінна
X = data_normalised[
    [
        "sqft_living",
        "sqft_lot",
        "sqft_above",
        "sqft_basement",
        "sqft_living15",
        "sqft_lot15",
    ]
]
y = data_normalised["price"]

# Розбиття на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Стандартизація даних
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Побудова моделі регресії з l2 регуляризацією
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation="relu", kernel_regularizer=l2(0.01)))  # Changed to linear activation for regression
model.compile(
    loss=lambda y_true, y_pred: 10 * tf.keras.losses.mean_squared_error(y_true, y_pred),
    optimizer="adam",
    metrics=["mean_absolute_error"],
)

# Callback для збереження найкращої моделі
checkpoint = ModelCheckpoint(
    "best_regression_model.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
)

# Тренування моделі
model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[checkpoint])

# Тестування моделі на тестовому наборі
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss[0]}")
print(f"Test MAE: {test_loss[1]}")

# Збереження найкращої моделі
model.save("best_regression_model.keras")
