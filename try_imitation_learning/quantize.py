import torch
import torch.nn as nn
from model import Model

device = torch.device("cpu")
model = Model(action_dim=2, max_action=1.0)

state_dict = torch.load('model.pt', map_location=device)
model.load_state_dict(state_dict)
model.eval().to(device)

model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

torch.save(model.state_dict(), 'quantized.pt')
