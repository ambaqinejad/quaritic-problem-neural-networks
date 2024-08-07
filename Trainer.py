import torch

from model.QuadraticEquation import QuadraticEquation
from utils import plotter
from utils import dataset_generator
from torch.optim.sgd import SGD
from torch.nn import L1Loss


class Trainer:
    def __init__(self, device):
        torch.manual_seed(42)
        self.model = QuadraticEquation(device)
        self.optimizer = SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = L1Loss()
        self.train_data, self.train_labels, self.test_data, self.test_labels = dataset_generator.generate(device)
        self.epoch_counts = []
        self.train_losses = []
        self.test_losses = []
        self.num_of_epochs = 1300

    def primary_plot(self):
        y_pred = self.model.forward(self.test_data)
        plotter.plot_predictions(self.train_data, self.train_labels, self.test_data, self.test_labels, y_pred)

    def train(self):
        torch.manual_seed(42)
        for epoch in range(self.num_of_epochs):
            self.model.train()
            y_pred = self.model.forward(self.train_data)
            train_loss = self.loss_fn(y_pred, self.train_labels)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch} ------> Train Loss: {train_loss:.4f}")
                self.model.eval()
                with torch.inference_mode():
                    test_pred = self.model.forward(self.test_data)
                    test_loss = self.loss_fn(test_pred, self.test_labels)
                    self.epoch_counts.append(epoch)
                    self.train_losses.append(train_loss)
                    self.test_losses.append(test_loss)
        plotter.plot_loss(self.epoch_counts, self.train_losses, self.test_losses)
