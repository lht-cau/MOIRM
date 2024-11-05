import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


# Convert to tensors and move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
dataset = TensorDataset(x_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size=88, shuffle=True)

# Define adjacency matrix calculation
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
    return g

def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine"
    dist = cosine_distance_torch(data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    
    adj = 1 - dist
    adj = adj * g
    adj_T = adj.transpose(0, 1)
    
    I = torch.eye(adj.shape[0]).to(data.device)
    
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    
    return adj.to_sparse()

# Define GCN layers
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).to(device))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features).to(device))
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

# Define the encoder
class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout, device):
        super(GCN_E, self).__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0], device)
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1], device)
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2], device)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)
        return x

# Define the regressor
class Regressor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Regressor, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_normal_(self.regressor[0].weight)

    def forward(self, x):
        return self.regressor(x)

# Initialize the model
def init_model_dict(num_view, dim_list, dim_he_list, output_dim, gcn_dropout=0.1, device='cuda'):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = GCN_E(dim_list[i], dim_he_list, gcn_dropout, device)
        model_dict["C{:}".format(i + 1)] = Regressor(dim_he_list[-1], output_dim).to(device)
    return model_dict

# Initialize optimizer
def init_optim(num_view, model_dict, lr=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)] = torch.optim.Adam(
            list(model_dict["E{:}".format(i + 1)].parameters()) + list(model_dict["C{:}".format(i + 1)].parameters()), 
            lr=lr)
    return optim_dict

# Model setup
dim_list = [RNA.shape[1], DNA.shape[1]]
dim_he_list = [1024, 512, 64]
output_dim = 1
model_dict = init_model_dict(2, dim_list, dim_he_list, output_dim, device=device)
optim_dict = init_optim(2, model_dict)

# Train the model
epochs = 2000  
best_mse = float('inf')

for epoch in range(epochs):
    model_dict["E1"].train()
    model_dict["E2"].train()
    
    for batch_X, batch_y in data_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        adj_rna = gen_adj_mat_tensor(batch_X[:, :dim_list[0]], parameter=0.9).to(device)
        adj_dna = gen_adj_mat_tensor(batch_X[:, dim_list[0]:], parameter=0.9).to(device)
        
        # Forward pass through RNA and DNA GCNs
        rna_feature = model_dict["E1"](batch_X[:, :dim_list[0]], adj_rna)
        dna_feature = model_dict["E2"](batch_X[:, dim_list[0]:], adj_dna)
       
        # Regressor for RNA and DNA
        rna_pred = model_dict["C1"](rna_feature)
        dna_pred = model_dict["C2"](dna_feature)
        
        # Combine predictions
        pred = (rna_pred + dna_pred) / 2
        loss = F.mse_loss(pred, batch_y)

        optim_dict["C1"].zero_grad()
        optim_dict["C2"].zero_grad()
        loss.backward()
        optim_dict["C1"].step()
        optim_dict["C2"].step()
    
    # Validation MSE calculation
    model_dict["E1"].eval()
    model_dict["E2"].eval()
    with torch.no_grad():
        outputs = (rna_pred + dna_pred) / 2
        mse = mean_squared_error(batch_y.cpu().numpy(), outputs.cpu().numpy())
        r2 = r2_score(batch_y.cpu().numpy(), outputs.cpu().numpy())
    
    if mse < best_mse:
        best_mse = mse
        torch.save(model_dict, "best_model.pth")
    
    print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse}, R2: {r2}")

print(f"Best MSE: {best_mse}")





import torch
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


best_model_path = 'best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_dict = torch.load(best_model_path)


for key in model_dict:
    model_dict[key].to(device)


model_dict["E1"].eval()
model_dict["E2"].eval()


x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)


all_results = []
COLS = list(X1.columns)


REPEAT_TIMES = 100


for _ in tqdm(range(REPEAT_TIMES)):
    results = []

    for k in range(1, len(COLS)):  
   
        save_col = x[:, k-1].copy()

   
        np.random.shuffle(x[:, k-1])
        x_tensor_shuffled = torch.tensor(x, dtype=torch.float32).to(device)
        
     
        adj_rna = gen_adj_mat_tensor(x_tensor_shuffled[:, :dim_list[0]], parameter=0.9).to(device)
        adj_dna = gen_adj_mat_tensor(x_tensor_shuffled[:, dim_list[0]:], parameter=0.9).to(device)
        
        with torch.no_grad():
           
            rna_feature = model_dict["E1"](x_tensor_shuffled[:, :dim_list[0]], adj_rna)
            dna_feature = model_dict["E2"](x_tensor_shuffled[:, dim_list[0]:], adj_dna)
            
           
            rna_pred = model_dict["C1"](rna_feature)
            dna_pred = model_dict["C2"](dna_feature)
            
           
            oof_preds = (rna_pred + dna_pred) / 2
            
            
            mae_score = mean_absolute_error(y, oof_preds.cpu().numpy())
            results.append(mae_score)
        
     
        x[:, k-1] = save_col

    all_results.append(results)


avg_results = np.mean(all_results, axis=0)


avg_results_df = pd.DataFrame({'feature': COLS[1:], 'avg_mae': avg_results})
avg_results_df = avg_results_df.sort_values('avg_mae', ascending=False)
print(avg_results_df)


avg_results_df.to_csv('hum_RFE_feature_importance.csv', index=False),


