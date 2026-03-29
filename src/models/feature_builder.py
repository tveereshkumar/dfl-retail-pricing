import torch

def build_features(base_features, prices, promos):

    if not isinstance(base_features, torch.Tensor):
        base_features = torch.tensor(base_features).float()

    if not isinstance(prices, torch.Tensor):
        prices = torch.tensor(prices).float()

    if not isinstance(promos, torch.Tensor):
        promos = torch.tensor(promos).float()

    prices = prices.unsqueeze(1)
    promos = promos.unsqueeze(1)

    # ✅ Correct concat
    features = torch.cat([base_features, prices, promos], dim=1)

    return features