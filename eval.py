import torch

from model import DigitClassifier

model = DigitClassifier.load_from_checkpoint("model.ckpt")

model.eval()

out = model(torch.rand(1, 3, 28, 28))
pred = torch.argmax(out, dim=1)

print(pred.item())