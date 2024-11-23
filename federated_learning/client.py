from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from task import load_dataset, load_model

class Client:
    def __init__(self, model, weights, X_train, y_train, X_test, y_test, learning_rate=0.01, batch_size=32, epochs=1, train_ratio=0.8, random_state=42):
        self.model = model
        self.model.set_weights(weights)
        self.train_ratio = train_ratio
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, train_size=train_ratio, random_state=random_state)
        self.X_test = X_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        # Train the model using the weights and the training data
        if self.train_ratio < 1:
            history = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.X_val, self.y_val))
        else:
            history = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size)

        return history

    def evaluate(self):
        # Evaluate the model using the test data
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, accuracy
    
    def get_weights(self):
        return self.model.get_weights()
    

if __name__ == "__main__":
    model = load_model()
    model.load_weights('weights\model.weights.h5')

    weights = model.get_weights()
    X_train, y_train, X_test, y_test = load_dataset()

    client = Client(
        model=model,
        weights=weights,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        learning_rate=0.01,
        batch_size=32,
        epochs=10,
        train_ratio=0.8,
        random_state=42
    )

    history = client.train()
    loss, accuracy = client.evaluate()
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    print("***** End *****")