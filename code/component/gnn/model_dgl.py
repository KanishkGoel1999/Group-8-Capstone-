import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class HeteroRGCNLayer(nn.Module):
    def __init__(self, hidden_size, out_size, etypes):
        """
        Defines a single layer of a Heterogeneous Relational Graph Convolutional Network (HeteroRGCN).
        """
        super(HeteroRGCNLayer, self).__init__()

        # Creates a separate weight matrix for each edge type stored in a nn.ModuleDict
        self.weight = nn.ModuleDict({etype: nn.Linear(hidden_size, out_size) for _, etype, _ in etypes})
        print("self.weight:", self.weight)
        # Print output: (merchant<>transaction): Linear(in_features=64, out_features=64, bias=True)

    def forward(self, G, feat_dict):
        """
        Performs the forward pass of the HeteroRGCN layer.
        """
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            if srctype in feat_dict:

                # feat_dict[srctype]: input feature matrix Nx64 for node type
                # self.weight[etype]: This is a relation-specific weight matrix for edge type etype
                # Wh is linear transformation
                Wh = self.weight[etype](feat_dict[srctype])
                #print(type(Wh))
                #Wh = feat_dict[srctype]
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.nodes[ntype].data}


# Model architecture
class HeteroRGCN(nn.Module):
    def __init__(self, ntype_dict, etypes, in_size_dict, hidden_size, out_size, n_layers, target_node):
        """
        Defines a Heterogeneous Relational Graph Convolutional Network (HeteroRGCN).
        """
        super(HeteroRGCN, self).__init__()

        # Input projection layers to ensure consistent feature size across node types Since nodes of different types
        # may have different feature dimensions, input_proj ensures that all node types have embeddings of size
        # hidden_size
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(in_size_dict[ntype], hidden_size) for ntype in in_size_dict
        })

        self.layers = nn.ModuleList()

        # 1st layer
        self.layers.append(HeteroRGCNLayer(hidden_size, hidden_size, etypes))

        # More layers
        for _ in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(hidden_size, hidden_size, etypes))

        # Final linear layer to map output target
        self.lin = nn.Linear(hidden_size, out_size)

        # g is graph, features_dict is dictionary with initial features, target_node is node type we are predicting
        # "transaction"
    def forward(self, g, features_dict, target_node):
        """
        Performs the forward pass of the HeteroRGCN model.
        """
        # Apply input projections to ensure all features are of size hidden_size
        # Ensures that all node types have feature vectors of the same size before message passing.
        x_dict = {ntype: self.input_proj[ntype](features_dict[ntype]) for ntype in features_dict}
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_dict = {k: F.leaky_relu(x) for k, x in x_dict.items()}

            # Passes the embeddings through all HeteroRGCN layers.
            x_dict = layer(g, x_dict)

        return self.lin(x_dict[target_node])
