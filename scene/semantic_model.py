import torch


class FeatureNorm(torch.nn.Module):
    def __init__(self):
        super(FeatureNorm, self).__init__()

    def forward(self, x):
        # assert len(x.shape) == 2
        return x / x.norm(dim=-1, keepdim=True)


class SemanticModel(torch.nn.Module):
    def __init__(self, dim_in=64, dim_hidden=128, dim_out=40, num_layer=3, device="cuda", use_bias=False, norm=False):
        super(SemanticModel, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layer = num_layer
        self.device = device
        self.args = {
            "dim_in": dim_in,
            "dim_hidden": dim_hidden,
            "dim_out": dim_out,
            "num_layer": num_layer,
            "device": device,
            "use_bias": use_bias,
            "norm": norm
        }
        layers = []
        for ind in range(num_layer):
            is_first = ind == 0
            # layer_w0 = w0_initial if is_first else w0

            layer_dim_in = dim_in if is_first else dim_hidden
            layer_dim_out = dim_out if ind == num_layer - 1 else dim_hidden
            layer = torch.nn.Linear(layer_dim_in, layer_dim_out, device=device, bias=use_bias)
            activation = torch.nn.ReLU() if ind < num_layer - 1 \
                else (torch.nn.Identity() if not norm else FeatureNorm()) #Softmax(dim=1)
            torch.nn.init.xavier_uniform_(layer.weight.data)

            layers.extend([layer, activation])
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, semantic_features):
        # shape = semantic_features.shape[-1]
        # semantic_features = semantic_features.view(-1, self.dim_in)
        semantic_labels = self.layers(semantic_features)
        # semantic_labels = semantic_labels.view(-1, shape, self.dim_out)
        return semantic_labels

    @staticmethod
    def load(path):
        pth = torch.load(path)
        model = SemanticModel(**pth["args"])
        model.load_state_dict(pth["state_dict"])
        return model

    def save(self, path):
        torch.save({
            "args": self.args,
            "state_dict": self.state_dict()
        }, path)

