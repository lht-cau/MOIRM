######Model training based on optimal hyperparameter combination and feature importance ranking based on feature permutation#####

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.base import BaseEstimator, RegressorMixin
import logging

from torch.utils.data import DataLoader, TensorDataset

####Set the corresponding number and layers of neurons based on the structure of the training dataset###

class CustomModel(nn.Module):
    def __init__(self, input_dim, neurons=2048, dropout_rate=0.2):
        super(CustomModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.inp1_size = RNA.shape[1]
        self.inp2_size = input_dim - RNA.shape[1]
        
        self.EnDNA = nn.Sequential(
            nn.Linear(self.inp1_size, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate)
        )
        
        self.EnRNA = nn.Sequential(
            nn.Linear(self.inp2_size, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate)
        )
        
        self.MO = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.Pred = nn.Sequential(
            nn.Linear(2048, neurons),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(neurons, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        inp1 = x[:, :self.inp1_size]
        inp2 = x[:, self.inp1_size:]
        
        EnDNA_out = self.EnDNA(inp1)
        EnRNA_out = self.EnRNA(inp2)
        
        MO_out = torch.cat((EnDNA_out, EnRNA_out), dim=1)
        MO_out = self.MO(MO_out)
        
        out = self.Pred(MO_out)
        return out


input_dim = x.shape[1]

####Set the optimal hyperparameter combination based on the results of cross validation and grid search####

neurons = 2048
dropout_rate = 0.2
learn_rate = 0.001
batch_size = 88
epochs = 8000 

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 
dataset = TensorDataset(x_tensor, y_tensor)


data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CustomModel(input_dim=input_dim, neurons=neurons, dropout_rate=dropout_rate)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=learn_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_mse = float('inf')
best_r2 = float('-inf')
best_model_path = 'best_MOIRM612_model.pth'

for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in data_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()


    model.eval()
    with torch.no_grad():
        outputs = model(x_tensor.to(device))
        outputs = outputs.cpu().numpy()
        mse = mean_squared_error(y, outputs)
        r2 = r2_score(y, outputs)

    print(f"Epoch {epoch+1}/{epochs}, MSE: {mse}, R2: {r2}")


    if mse < best_mse:
        best_mse = mse
        best_r2 = r2
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved!")

print(f"Best MSE: {best_mse}, Best R2: {best_r2}")


from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

best_model = CustomModel(input_dim=input_dim, neurons=neurons, dropout_rate=dropout_rate)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)


best_model.eval()
all_results = []
COLS = list(X1.columns)

REPEAT_TIMES = 100

for _ in tqdm(range(REPEAT_TIMES)):
    results = []

    for k in range(len(COLS)):
        if k > 0:
            save_col = x[:, k-1].copy()
            np.random.shuffle(x[:, k-1])
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                
            with torch.no_grad():
                oof_preds = best_model(x_tensor).cpu().numpy()
                
            mae_score = mean_absolute_error(y, oof_preds)
            results.append(mae_score)
            
           
            x[:, k-1] = save_col

    all_results.append(results)


avg_results = np.mean(all_results, axis=0)


avg_results_df = pd.DataFrame({'feature': COLS[1:], 'avg_mae': avg_results})
avg_results_df = avg_results_df.sort_values('avg_mae', ascending=False)
print(avg_results_df)

avg_results_df.to_csv('pig_feature_importance.csv', index=False)


