from concurrent.futures import ThreadPoolExecutor
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
        self.train_dataset, self.test_dataset = load_dataset(batch_size=self.batch_size)
        self.dataset_list = self.split_dataset()

    def get_weights(self):
        return self.model.get_weights()

    def split_dataset(self):
        # Split dataset into chunks for each client
        return list(self.train_dataset.as_numpy_iterator())

    def modify_server_weights(self, client_weights_list):
        averaged_weights = []
        for i in range(len(client_weights_list[0])):
            stacked_weights = np.stack([weights[i] for weights in client_weights_list])
            averaged_weight = np.mean(stacked_weights, axis=0)
            averaged_weights.append(averaged_weight)

        return averaged_weights

    def evaluate(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_dataset)
        return test_loss, test_accuracy

    def create_and_train_client(self, client_id, dataset):
        print(f"############# Client {client_id + 1} ###############")
        client = Client(
            model=self.model,
            weights=self.model.get_weights(),
            dataset=dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )
        history = client.train()
        test_loss, test_accuracy = client.evaluate()
        print(f"Client {client_id + 1} Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")
        return client.get_weights()

    def run(self):
        for round in range(1, self.num_rounds + 1):
            print(f"**************[ ROUND : {round} ]********************")
            client_weights_list = []

            with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
                futures = [
                    executor.submit(
                        self.create_and_train_client,
                        client_id,
                        dataset
                    )
                    for client_id, dataset in enumerate(self.dataset_list[:self.num_clients])
                ]

                for future in futures:
                    client_weights = future.result()
                    client_weights_list.append(client_weights)

            new_server_weights = self.modify_server_weights(client_weights_list)
            self.model.set_weights(new_server_weights)

            test_loss, test_accuracy = self.evaluate()
            print(f"Server Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    server = Server(
        num_rounds=10,
        num_clients=5,
        local_epochs=10,
        batch_size=32,
        learning_rate=0.01,
    )
    server.run()
