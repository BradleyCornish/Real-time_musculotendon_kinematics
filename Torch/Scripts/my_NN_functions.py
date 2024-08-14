import pandas
import torch
import torch.nn as nn
import pickle


def load_data(data_dir,DOFtype):
    fp="/"
    features,labels={},{}
    for split in ["Train","Val","Test"]:
        feature_data = pandas.read_csv(data_dir+fp+split+fp+DOFtype+'_features.csv')
        label_data = pandas.read_csv(data_dir+fp+split+fp+DOFtype+'_labels.csv')
        features[split] = feature_data[[col for col in feature_data.columns if 'Unnamed' not in col]].values
        labels[split] = label_data[[col for col in label_data.columns if 'Unnamed' not in col]].values
        if split == "Train":
            feature_columns = [col for col in feature_data.columns if 'Unnamed' not in col]
            labels_columns = [col for col in label_data.columns if 'Unnamed' not in col]
    return features,labels,feature_columns,labels_columns

# Function to save the HP tuning study
def save_study(study, filename):
    with open(filename, 'wb') as f:
        pickle.dump(study, f)

# Function to load the HP tuning study
def load_study(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim):
        """
        Args:
            input_dim (int): Dimension of the input features.
            layer_dims (list of int): List containing the number of neurons for each hidden layer.
            output_dim (int): Dimension of the output layer.
        """
        super(FeedForwardNN, self).__init__()
        
        # Create a list to hold all the layers
        layers = []
        
        # Input layer to first hidden layer
        if layer_dims:
            layers.append(nn.Linear(input_dim, layer_dims[0]))
            layers.append(nn.ReLU())
        
        # Add the subsequent hidden layers
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
            layers.append(nn.ReLU())
        
        # Last hidden layer to output layer
        if layer_dims:
            layers.append(nn.Linear(layer_dims[-1], output_dim))
        else:
            layers.append(nn.Linear(input_dim, output_dim))
        
        # Use the Sequential container to combine the layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# Define the model
# class FeedForwardNN_HP(nn.Module):
#     def __init__(self, input_dim, layer_dims, output_dim, activation_funcs):
#         super(FeedForwardNN_HP, self).__init__()
#         layers = []
#         in_dim = input_dim
#         for dim, activation_func in zip(layer_dims, activation_funcs):
#             layers.append(nn.Linear(in_dim, dim))
#             if activation_func == 'relu':
#                 layers.append(nn.ReLU())
#             elif activation_func == 'sigmoid':
#                 layers.append(nn.Sigmoid())
#             elif activation_func == 'tanh':
#                 layers.append(nn.Tanh())
#             in_dim = dim
#         layers.append(nn.Linear(in_dim, output_dim))
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)
    
class FeedForwardNN_HP(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim, activation_funcs, l1_lambdas, l2_lambdas):
        super(FeedForwardNN_HP, self).__init__()
        layers = []
        in_dim = input_dim
        for dim, activation_func in zip(layer_dims, activation_funcs):
            layers.append(nn.Linear(in_dim, dim))
            if activation_func == 'relu':
                layers.append(nn.ReLU())
            elif activation_func == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation_func == 'tanh':
                layers.append(nn.Tanh())
            in_dim = dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self.l1_lambdas = l1_lambdas
        self.l2_lambdas = l2_lambdas

    def forward(self, x):
        return self.layers(x)
    
    def compute_regularization_loss(self):
        l1_loss = 0.0
        l2_loss = 0.0
        for i, layer in enumerate(self.l1_lambdas):
            if isinstance(layer, nn.Linear):
                l1_loss += self.l1_lambdas[i] * torch.sum(torch.abs(layer.weight))
                l2_loss += self.l2_lambdas[i] * torch.sum(layer.weight ** 2)
        return l1_loss + l2_loss

