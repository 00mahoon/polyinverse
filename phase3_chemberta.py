import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')

class PolymerDataset(Dataset):
    def __init__(self, smiles_list, density_list, tc_list, max_length=128):
        self.smiles = smiles_list
        self.density = density_list
        self.tc = tc_list
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.smiles[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'density': torch.tensor(self.density[idx], dtype=torch.float),
            'tc': torch.tensor(self.tc[idx], dtype=torch.float),
        }

class ChemBERTaRegressor(nn.Module):
    def __init__(self):
        super(ChemBERTaRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        hidden_size = self.bert.config.hidden_size
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head_density = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_tc = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        shared = self.shared(cls)
        return self.head_density(shared).squeeze(), self.head_tc(shared).squeeze()

public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')
df = pd.concat([public, private], ignore_index=True)
df = df[df['Density'].notna() | df['Tc'].notna()].reset_index(drop=True)
print(f"Total samples: {len(df)}")

density_mean = df['Density'].mean()
density_std = df['Density'].std()
tc_mean = df['Tc'].mean()
tc_std = df['Tc'].std()

smiles_list, density_list, tc_list = [], [], []
for _, row in df.iterrows():
    d = (row['Density'] - density_mean) / density_std if pd.notna(row['Density']) else float('nan')
    t = (row['Tc'] - tc_mean) / tc_std if pd.notna(row['Tc']) else float('nan')
    smiles_list.append(row['SMILES'])
    density_list.append(d)
    tc_list.append(t)

train_idx, test_idx = train_test_split(range(len(smiles_list)), test_size=0.2, random_state=42)

train_dataset = PolymerDataset(
    [smiles_list[i] for i in train_idx],
    [density_list[i] for i in train_idx],
    [tc_list[i] for i in train_idx]
)
test_dataset = PolymerDataset(
    [smiles_list[i] for i in test_idx],
    [density_list[i] for i in test_idx],
    [tc_list[i] for i in test_idx]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = ChemBERTaRegressor()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nFine-tuning ChemBERTa...")

best_r2_d = -999
for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        pred_d, pred_t = model(batch['input_ids'], batch['attention_mask'])

        loss = torch.tensor(0.0, requires_grad=True)
        mask_d = ~torch.isnan(batch['density'])
        mask_t = ~torch.isnan(batch['tc'])

        if mask_d.sum() > 0:
            loss = loss + ((pred_d[mask_d] - batch['density'][mask_d]) ** 2).mean()
        if mask_t.sum() > 0:
            loss = loss + ((pred_t[mask_t] - batch['tc'][mask_t]) ** 2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        d_true, d_pred, t_true, t_pred = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                pd_out, pt_out = model(batch['input_ids'], batch['attention_mask'])
                mask_d = ~torch.isnan(batch['density'])
                mask_t = ~torch.isnan(batch['tc'])
                if mask_d.sum() > 0:
                    d_true.extend((batch['density'][mask_d].numpy() * density_std + density_mean).tolist())
                    d_pred.extend((pd_out[mask_d].numpy() * density_std + density_mean).tolist())
                if mask_t.sum() > 0:
                    t_true.extend((batch['tc'][mask_t].numpy() * tc_std + tc_mean).tolist())
                    t_pred.extend((pt_out[mask_t].numpy() * tc_std + tc_mean).tolist())

        r2_d = r2_score(d_true, d_pred) if len(d_true) > 1 else 0
        r2_t = r2_score(t_true, t_pred) if len(t_true) > 1 else 0
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:>2}/50 | Loss: {avg_loss:.4f} | Density R²: {r2_d:.3f} | Tc R²: {r2_t:.3f}")

        if r2_d > best_r2_d:
            best_r2_d = r2_d
            torch.save(model.state_dict(), 'best_chemberta.pt')

model.load_state_dict(torch.load('best_chemberta.pt'))
model.eval()
d_true, d_pred, t_true, t_pred = [], [], [], []
with torch.no_grad():
    for batch in test_loader:
        pd_out, pt_out = model(batch['input_ids'], batch['attention_mask'])
        mask_d = ~torch.isnan(batch['density'])
        mask_t = ~torch.isnan(batch['tc'])
        if mask_d.sum() > 0:
            d_true.extend((batch['density'][mask_d].numpy() * density_std + density_mean).tolist())
            d_pred.extend((pd_out[mask_d].numpy() * density_std + density_mean).tolist())
        if mask_t.sum() > 0:
            t_true.extend((batch['tc'][mask_t].numpy() * tc_std + tc_mean).tolist())
            t_pred.extend((pt_out[mask_t].numpy() * tc_std + tc_mean).tolist())

r2_d = r2_score(d_true, d_pred)
r2_t = r2_score(t_true, t_pred)
mae_d = mean_absolute_error(d_true, d_pred)
mae_t = mean_absolute_error(t_true, t_pred)

print(f"\n=== ChemBERTa Best Performance ===")
print(f"Density — R²: {r2_d:.3f} | MAE: {mae_d:.4f} g/ml")
print(f"Tc      — R²: {r2_t:.3f} | MAE: {mae_t:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(d_true, d_pred, alpha=0.5, color='steelblue')
axes[0].plot([min(d_true), max(d_true)], [min(d_true), max(d_true)], 'r--')
axes[0].set_title(f'Density (R²={r2_d:.3f})')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[1].scatter(t_true, t_pred, alpha=0.5, color='darkorange')
axes[1].plot([min(t_true), max(t_true)], [min(t_true), max(t_true)], 'r--')
axes[1].set_title(f'Tc (R²={r2_t:.3f})')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
plt.tight_layout()
plt.savefig('chemberta_results.png')
print("Saved: chemberta_results.png")
