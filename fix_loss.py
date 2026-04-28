# Fix masked_mse_loss function
import re

with open('phase2_multitask_gnn.py', 'r') as f:
    code = f.read()

old = '''def masked_mse_loss(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return ((pred[mask] - target[mask]) ** 2).mean()'''

new = '''def masked_mse_loss(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    loss = ((pred - target) ** 2)
    loss = loss * mask.float()
    return loss.sum() / mask.sum()'''

code = code.replace(old, new)

with open('phase2_multitask_gnn.py', 'w') as f:
    f.write(code)

print("Fixed!")
