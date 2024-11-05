#####Cross validation for hyperparameter selection######
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.base import BaseEstimator, RegressorMixin
import logging

####Set the corresponding number and layers of neurons based on the structure of the training dataset###

class CustomModel(nn.Module):
    def __init__(self, input_dim, neurons=512, dropout_rate=0.0):
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
            nn.Linear(self.inp2_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate)
        )
        
        self.MO = nn.Sequential(
            nn.BatchNorm1d(1152),
            nn.Linear(1152, 1152),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1152, 1152),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1152, 1152),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.Pred = nn.Sequential(
            nn.Linear(1152, neurons),
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

class PytorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, neurons=512, dropout_rate=0.0, learn_rate=0.01, epochs=10, batch_size=58, optimizer='adam'):
        self.neurons = neurons
        self.dropout_rate = dropout_rate
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 ######Divide the training set and testing set according to the appropriate proportion based on the dataset structure#####       

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        self.model = CustomModel(X.shape[1], neurons=self.neurons, dropout_rate=self.dropout_rate)
        self.model.to(self.device)
        
        criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False
        )

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()

            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            logging.info(f"Epoch {epoch+1}/{self.epochs}, Validation Loss: {val_loss}")
            print(f"Model is currently on device: {next(self.model.parameters()).device}")

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
        return predictions




####Set hyperparameters that need to be selected######


def train_and_evaluate(x, y):
    try:
        Regressor = PytorchRegressor()


        mse = make_scorer(mean_squared_error, greater_is_better=False)
        r2 = make_scorer(r2_score, greater_is_better=True)
        sc = {'mse': mse, 'r2': r2}

        batch_size = [29,58,87,116]
        epochs = [50,100,200,300,400]
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam']
        neurons = [64,128,512,1024,2048]
        dropout_rate = [0.0,0.01,0.05,0.1,0.2]
        learn_rate = [0.001,0.05,0.01,0.1,0.2]

        param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer,
                          neurons=neurons, dropout_rate=dropout_rate, learn_rate=learn_rate)

        

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        grid_search = GridSearchCV(
            estimator=Regressor,
            param_grid=param_grid,
            cv=kf,
            n_jobs=2,
            verbose=1,
            scoring=sc,
            refit='mse'
        )

        logging.info("Starting Grid Search...")
        grid_search.fit(x, y)

        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv('pig_result_teat.csv', index=False)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        logging.info(f"Best parameters found: {best_params}")
        logging.info(f"Best score: {best_score}")

        print("Best parameters found: ", best_params)
        print("Best score: ", best_score)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    x = X1.values
    y = PHE['BMI'].values
    train_and_evaluate(x, y)