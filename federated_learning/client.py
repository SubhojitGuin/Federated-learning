class Client:
    def __init__(self, model, weights, dataset, learning_rate=0.01, batch_size=32, epochs=1):
        self.model = model
        self.model.set_weights(weights)
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        # Train the model on the client's dataset
        history = self.model.fit(self.dataset, epochs=self.epochs)
        return history

    def evaluate(self):
        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.dataset)
        return loss, accuracy

    def get_weights(self):
        return self.model.get_weights()
