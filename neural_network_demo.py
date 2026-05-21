from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load_breast_cancer can return either a Bunch or (X, y) depending on usage.
# Use `return_X_y=True` to get a consistent (X, y) tuple and avoid type checker issues.
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Build the model with the expected input shape to avoid passing input kwargs to layers
model.build(input_shape=(None, X_train.shape[1]))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=20, batch_size=16)

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)