import torch

class KLLoss:
    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        batch_size = div.shape[0]
        return div.mean()
    
    def __repr__(self):
        return "KLLoss()"
    

class KLLossMulti:
    def __init__(self):
        self.kl_loss = KLLoss()

    def __call__(self, q_list, p_list):
        return sum([self.kl_loss(q, p) for q, p in zip(q_list, p_list)])
    
    def __repr__(self):
        return "KLLossMulti()"