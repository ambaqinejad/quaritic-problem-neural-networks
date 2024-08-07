from matplotlib import pyplot as plt


def plot_predictions(train_data=None, train_labels=None, test_data=None, test_labels=None, predictions=None):
    plt.figure(figsize=(10, 7))
    train_data = train_data.cpu().detach().numpy()
    train_labels = train_labels.cpu().detach().numpy()
    test_data = test_data.cpu().detach().numpy()
    test_labels = test_labels.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()
    plt.scatter(train_data, train_labels, c='b', s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c='r', s=4, label="Test Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="g", s=4, label="Predictions")

    plt.legend()
    plt.show()


def plot_loss(epoch_count, train_loss_values, test_loss_values):
    plt.figure(figsize=(10, 7))
    train_loss_values = [tensor.cpu().detach().numpy() for tensor in train_loss_values]
    test_loss_values = [tensor.cpu().detach().numpy() for tensor in test_loss_values]
    plt.plot(epoch_count, train_loss_values, label="Training Loss")
    plt.plot(epoch_count, test_loss_values, label="Testing Loss")
    plt.title("Training Loss and Testing Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
