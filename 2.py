import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Побудова та тренування моделі для кожного значення dropout
for dropout_rate in [0, 0.3, 0.5, 0.9]:
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))  # Changed to linear activation for regression
    model.compile(
        loss="mean_squared_error",  # Changed loss function for regression
        optimizer="adam",
        metrics=["mean_absolute_error"],  # Changed metric for regression
    )

    # Callback для TensorBoard
    tensorboard = TensorBoard(
        log_dir=f"./logs_dropout_{dropout_rate}",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )

    # Тренування моделі
    model.fit(
        X_train,
        y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[tensorboard],
    )

    # Тестування моделі на тестовому наборі
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Dropout Rate: {dropout_rate}, Test MAE: {test_mae}, Test Loss: {test_loss}")

    # Збереження моделі
    model.save(f"model_dropout_{dropout_rate}.keras")
