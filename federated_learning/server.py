from concurrent.futures import ThreadPoolExecutor
from task import load_dataset, load_model
from client import Client
import numpy as np
import os
from utils import newloggingfunction
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

RESULT_FOLDER_PATH = "results"
LOG_FOLDER_PATH = "log"

LABEL_FLIP = {
    0: 6,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 0,
    7: 7,
    8: 8,
    9: 9,
}

class Server:
    def __init__(self, num_rounds=1, num_clients=30, local_epochs=1, batch_size=1, learning_rate=0.01, poisoned_clients=0):
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = load_model(self.learning_rate)
        self.X_train, self.y_train, self.X_test, self.y_test = load_dataset()
        self.client_counter = 0
        self.poisoned_clients = poisoned_clients

    def get_weights(self):
        return self.model.get_weights()
    
    def dataset_flip_label(self, X_train, y_train):
        for i in range(len(y_train)):
            y_train[i] = LABEL_FLIP[y_train[i]]

        return X_train, y_train

    def create_and_train_client(self, client_id, X_train, y_train, X_test, y_test):
        print(f"\n############# Client {client_id + 1} ###############\n")

        if (client_id < self.poisoned_clients):
            X_train, y_train = self.dataset_flip_label(X_train, y_train)
            print("%%%%%%% Poisoned %%%%%%%\n")

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

        # print("Available Metrics:", history.history.keys())
        # print("Training accuracy:", history.history['accuracy'])
        # print("Validation Accuracy:", history.history['val_accuracy'])
        # print("\n")
        # print("Training Loss:", history.history['loss'])
        # print("Validation Loss:", history.history['val_loss'])
        # print("\n")

        # Get the testing accuracy
        test_loss, test_accuracy = client.evaluate()
        print("Test Accuracy:", test_accuracy)
        print("Test Loss:", test_loss)

        # Get the model weights
        client_weights = client.get_weights()
        return client_weights

    def split_dataset(self):
        X_train, y_train = self.X_train, self.y_train
        num_clients = self.num_clients
        dataset_list = []
        len_train = X_train.shape[0] // num_clients

        for i in range(num_clients):
            start = i * len_train
            end = (i + 1) * len_train
            subset_images = X_train[start:end]
            subset_labels = y_train[start:end]
            dataset_list.append((subset_images, subset_labels))

        return dataset_list

    def check_valid_weights(self, client_weights):
        return True

    def modify_server_weights(self, client_weights_list):
        averaged_weights = []
        for i in range(len(client_weights_list[0])):
            stacked_weights = np.stack([weights[i] for weights in client_weights_list])
            averaged_weight = np.mean(stacked_weights, axis=0)
            averaged_weights.append(averaged_weight)

        return averaged_weights

    def evaluate(self):
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test)
        return test_loss, test_accuracy
    
    def execute_round(self, dataset_list):
        client_weights_list = []

        # # Use ThreadPoolExecutor to run clients in parallel
        # with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
        #     futures = [
        #         executor.submit(
        #             self.create_and_train_client,
        #             client_id,
        #             dataset_list[client_id][0],
        #             dataset_list[client_id][1],
        #             self.X_test,
        #             self.y_test,
        #         )
        #         for client_id in range(self.num_clients)
        #     ]

        #     # Collect results as they complete
        #     for future in futures:
        #         client_weights = future.result()
        #         if self.check_valid_weights(client_weights=client_weights):
        #             client_weights_list.append(client_weights)

        for client_id in range(self.num_clients):
            client_weights = self.create_and_train_client(client_id, dataset_list[client_id][0], dataset_list[client_id][1], self.X_test, self.y_test)
            if self.check_valid_weights(client_weights):
                client_weights_list.append(client_weights)

        new_server_weights = self.modify_server_weights(client_weights_list)
        self.model.set_weights(new_server_weights)

        # Evaluate after each round
        test_loss, test_accuracy = self.evaluate()

        return test_loss, test_accuracy

    def run(self):
        dataset_list = self.split_dataset()

        for round in range(1, self.num_rounds + 1):
            print(f"\n\n**************[ ROUND : {round} ]********************\n\n")
            test_loss, test_accuracy = self.execute_round(dataset_list)

            FileResult = open(f"{RESULT_FOLDER_PATH}/{category}_{date}__results.csv", "a")
            FileResult.write(f"{round}|{test_accuracy}|{test_loss}\n")
            FileResult.close()

            print("\n****************************\n")
            print("Server Test Loss:", test_loss)
            print("Server Test Accuracy:", test_accuracy)
            print("\n****************************")
            

if __name__ == "__main__":
    category = "benign"
    date = datetime.now().strftime("%Y-%m-%d__%H_%M")

    # Create log folder
    if os.path.exists(LOG_FOLDER_PATH) and os.path.isdir(LOG_FOLDER_PATH):
        print(f"{LOG_FOLDER_PATH} exists")
    else:
        os.makedirs(LOG_FOLDER_PATH)
        print(f"{LOG_FOLDER_PATH} created")
    
    # print = newloggingfunction(category, datetime.now().strftime("%Y-%%m-%d__%H_%M")

    # Create RESULT folder
    if os.path.exists(RESULT_FOLDER_PATH) and os.path.isdir(RESULT_FOLDER_PATH):
        print(f"{RESULT_FOLDER_PATH} exists")
    else:
        os.makedirs(RESULT_FOLDER_PATH)
        print(f"{RESULT_FOLDER_PATH} created")

    FileResult = open(f"{RESULT_FOLDER_PATH}/{category}_{date}__results.csv", "w")
    FileResult.write("ROUND|TEST ACCURACY|TEST LOSS\n")
    FileResult.close()
    

    server = Server(
        num_rounds=50,
        num_clients=30,
        local_epochs=2,
        batch_size=32,
        learning_rate=0.01,
        poisoned_clients=0
    )

    server.run()
