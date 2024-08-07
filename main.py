import torch
from Trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(device=device)
trainer.primary_plot()
trainer.train()
trainer.primary_plot()
trainer.save_model()