from utils import plotter, dataset_generator
from model.QuadraticEquation import QuadraticEquation
import torch
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data, train_labels, test_data, test_labels = dataset_generator.generate(device)

model = QuadraticEquation(device)
path = Path("./models_dir")
model_name = "quadratic_equation_model_0.pth"
model_path = path / model_name

model.load_state_dict(torch.load(f=model_path))
y_pred = model.forward(test_data)
plotter.plot_predictions(train_data, train_labels, test_data, test_labels, y_pred)