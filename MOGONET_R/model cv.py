
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
import logging


# Convert to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
dataset = TensorDataset(x_tensor, y_tensor)

###recording
logging.basicConfig(filename='model_training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Split data for training/validation
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Define adjacency matrix calculation functions
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def graph_from_dist_tensor(dist, parameter, self_dist=True):
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
    return g

def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    dist = cosine_distance_torch(data)
    g = graph_from_dist_tensor(dist, parameter)
    adj = 1 - dist
    adj = adj * g
    adj_T = adj.t()
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

# Define full model and optimizer initialization
def init_model_dict(num_view, dim_list, dim_he_list, output_dim, gcn_dropout=0.5, device='cuda'):
    model_dict = {}
    for i in range(num_view):
        model_dict[f"E{i + 1}"] = GCN_E(dim_list[i], dim_he_list, gcn_dropout, device)
        model_dict[f"C{i + 1}"] = Regressor(dim_he_list[-1], output_dim).to(device)
    return model_dict

def init_optim(num_view, model_dict, lr=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict[f"C{i + 1}"] = torch.optim.Adam(
            list(model_dict[f"E{i + 1}"].parameters()) + list(model_dict[f"C{i + 1}"].parameters()), 
            lr=lr)
    return optim_dict

# Define Pytorch-based estimator class for GridSearchCV#
#Convert the classification model into a regression model and remove the VCDN module, which is mainly used for optimizing classification tasks###
class PytorchGCNRegressor:
    def __init__(self, gcn_dropout=0.5, dim_he_list=[2048, 1024, 32], parameter=0.5, learn_rate=1e-4, batch_size=88, epochs=1000):
        self.gcn_dropout = gcn_dropout
        self.dim_he_list = dim_he_list
        self.parameter = parameter
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dict = None  

    def get_params(self, deep=True):
        return {
            'gcn_dropout': self.gcn_dropout,
            'dim_he_list': self.dim_he_list,
            'parameter': self.parameter,
            'learn_rate': self.learn_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        dim_list = [RNA.shape[1], DNA.shape[1]]  
        output_dim = 1  

        self.model_dict = init_model_dict(2, dim_list, self.dim_he_list, output_dim, gcn_dropout=self.gcn_dropout, device=self.device)
        optim_dict = init_optim(2, self.model_dict, lr=self.learn_rate)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).to(self.device), torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device))
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for batch_X, batch_y in data_loader:
                adj_rna = gen_adj_mat_tensor(batch_X[:, :dim_list[0]], parameter=self.parameter).to(self.device)
                adj_dna = gen_adj_mat_tensor(batch_X[:, dim_list[0]:], parameter=self.parameter).to(self.device)

                rna_feature = self.model_dict["E1"](batch_X[:, :dim_list[0]], adj_rna)
                dna_feature = self.model_dict["E2"](batch_X[:, dim_list[0]:], adj_dna)
                rna_pred = self.model_dict["C1"](rna_feature)
                dna_pred = self.model_dict["C2"](dna_feature)

                pred = (rna_pred + dna_pred) / 2 
                loss = F.mse_loss(pred, batch_y)

                optim_dict["C1"].zero_grad()
                optim_dict["C2"].zero_grad()
                loss.backward()
                optim_dict["C1"].step()
                optim_dict["C2"].step()

    def predict(self, X):
        self.model_dict["E1"].eval() 
        self.model_dict["E2"].eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            adj_rna = gen_adj_mat_tensor(X_tensor[:, :RNA.shape[1]], parameter=self.parameter).to(self.device)
            adj_dna = gen_adj_mat_tensor(X_tensor[:, RNA.shape[1]:], parameter=self.parameter).to(self.device)

            rna_feature = self.model_dict["E1"](X_tensor[:, :RNA.shape[1]], adj_rna)
            dna_feature = self.model_dict["E2"](X_tensor[:, RNA.shape[1]:], adj_dna)
            rna_pred = self.model_dict["C1"](rna_feature)
            dna_pred = self.model_dict["C2"](dna_feature)

            pred = (rna_pred + dna_pred) / 2

        return pred.cpu().numpy()
 

#### Define grid search parameters###

def train_and_evaluate(x, y):
    try:
        param_grid = {
        'gcn_dropout': [0.01,0.1, 0.2, 0.5],
                 'batch_size': [22,44,88],
                'epochs': [1000],
        'dim_he_list': [[1024, 512, 64], [2048, 1024, 128],[2048,512,32]],
        'parameter': [0.1,0.3, 0.5, 0.7,0.9]
             }

        model = PytorchGCNRegressor() 

        mse = make_scorer(mean_squared_error, greater_is_better=False)
        r2 = make_scorer(r2_score, greater_is_better=True)
        sc = {'mse': mse, 'r2': r2}
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(model, param_grid, cv=kf, n_jobs=2, verbose=1, scoring=sc,refit='mse')
        grid_search.fit(x, y)

        logging.info("Starting Grid Search...")
        grid_search.fit(x, y)

        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv('result_teat.csv', index=False)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        logging.info(f"Best parameters found: {best_params}")
        logging.info(f"Best score: {best_score}")

        print("Best parameters found: ", grid_search.best_params_)
        print("Best score found: ", grid_search.best_score_)

        return grid_search

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")



train_and_evaluate(x, y)
