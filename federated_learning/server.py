from task import load_dataset, load_model
from client import Client
import numpy as np

class Server:
    def __init__(self, num_rounds=1, num_clients=30, local_epochs=1, batch_size=1, learning_rate=0.01):
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = load_model()
        self.X_train, self.y_train, self.X_test, self.y_test = load_dataset()
        self.client_counter = 0

    def get_weights(self):
        return self.model.get_weights()

    def create_and_train_client(self, X_train, y_train, X_test, y_test):
        self.client_counter += 1

        print(f"############# Client {self.client_counter} ###############")

        # Simulate client call
        client = Client(
            model=self.model, 
            weights=self.model.get_weights(), 
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            train_ratio=0.8,
            random_state=42
        )

        history = client.train()

        print("Aavailable Metrics:", history.history.keys())
        print("Training accuracy:", history.history['accuracy'])
        print("Validation Accuracy:", history.history['val_accuracy'])

        print("\n")

        print("Training Loss:", history.history['loss'])
        print("Validation Loss:", history.history['val_loss'])

        print("\n")

        # Get the testing accuracy
        test_loss, test_accuracy = client.evaluate()
        print("Test Accuracy:", test_accuracy)
        print("Test Loss:", test_loss)

        # Get the model weights
        client_weights = client.get_weights()

        return client_weights
    
    def split_dataset(self):
        # Split the training data for all the clients
        # Each client will have the same number of samples
        X_train, y_train = self.X_train, self.y_train
        num_clients = self.num_clients
        dataset_list = []
        len_train = X_train.shape[0]//num_clients

        for i in range(num_clients):
            start = i * len_train
            end = (i + 1) * len_train

            subset_images = X_train[start:end]
            subset_labels = y_train[start:end]

            train_data = (subset_images, subset_labels)

            dataset_list.append(train_data)

        return dataset_list
    
    def check_valid_weights(self, client_weights):
        return True
    
    def modify_server_weights(self, client_weights_list):
        averaged_weights = []
        for i in range(len(client_weights_list[0])):
            # Stack weights from all models for the current layer
            stacked_weights = np.stack([weights[i] for weights in client_weights_list])
            # Calculate the mean across all models
            averaged_weight = np.mean(stacked_weights, axis=0)
            averaged_weights.append(averaged_weight)

        return averaged_weights
    
    def evaluate(self):
        # Evaluate the model on the test data
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test)
        return test_loss, test_accuracy

    def run(self):
        # Split the dataset for all the clients
        dataset_list = self.split_dataset()

        for round in range(1, self.num_rounds + 1):
            self.client_counter = 0
            print(f"**************[ ROUND : {round} ]********************")

            client_weights_list = []
            for client in range(self.num_clients):
                # Get the client's dataset
                client_X_train, client_y_train = dataset_list[client]
                client_weights = self.create_and_train_client(client_X_train, client_y_train, self.X_test, self.y_test)
                if self.check_valid_weights(client_weights=client_weights):
                    client_weights_list.append(client_weights)

            new_server_weights = self.modify_server_weights(client_weights_list)
            self.model.set_weights(new_server_weights)

            # Evaluate after each round
            test_loss, test_accuracy = self.evaluate()

            print("Server Test Loss:", test_loss)
            print("Server Test Accuracy:", test_accuracy)

if __name__ == "__main__":
    server = Server(
        num_rounds=10,
        num_clients=5,
        local_epochs=10,
        batch_size=32,
        learning_rate=0.01,
    )

    server.run()