import numpy as np
import scipy
from torchmetrics import Metrics
import torch

def calculate_activation_statistics(activations):
    """
    Inputs:
        activations: samples x dim
    Returns:
        mu: dim
        sigma: dim x dim
    """

    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_fid(stat_1, stat_2, eps=1e-6):
    """ From https://github.com/ChenFengYe/motion-latent-diffusion/blob/main/mld/models/metrics/utils.py
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(stat_1[0])
    mu2 = np.atleast_1d(stat_2[0])

    sigma1 = np.atleast_2d(stat_1[1])
    sigma2 = np.atleast_2d(stat_2[1])

    assert (mu1.shape == mu2.shape
            ), "Training and test mean vectors have different lengths"
    assert (sigma1.shape == sigma2.shape
            ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
            # print("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean

class FID(Metrics):
    def __init__(self, config, **kwargs):
        super().__init__(dist_sync_on_step=config.dist_sync_on_step)
        self.name = 'fid'
        self.metrics = []

        self.embedding_model = None
        # self.embedding_model = EmbeddingModel(config.fid.model) #TODO: Add embedding model

        self.add_state('fid', default=torch.tensor(0.0), dist_reduce_fx='mean')
        self.add_state('fid_gt', default=torch.tensor(0.0), dist_reduce_fx='mean')
        self.add_state('recon_motion_embeddings', default=[], dist_reduce_fx=None)
        self.add_state('gt_motion_embeddings', default=[], dist_reduce_fx=None)

        self.metrics.extend(["fid", "fid_gt"])        

    def compute(self):
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # Concatenate all embeddings
        all_gt_embeddings = torch.cat(self.gt_motion_embeddings, axis=0)
        all_gt_embeddings2 = all_gt_embeddings.clone()[
            torch.randperm(all_gt_embeddings.shape[0]), :]
        all_recon_embeddings = torch.cat(self.recon_motion_embeddings, axis=0)
        
        gt_stat = calculate_activation_statistics(all_gt_embeddings)
        gt_stat2 = calculate_activation_statistics(all_gt_embeddings2)
        recon_stat = calculate_activation_statistics(all_recon_embeddings)

        # Compute FID
        fid = calculate_fid(gt_stat, recon_stat)
        fid_gt = calculate_fid(gt_stat, gt_stat2)

        metrics['fid'] = fid
        metrics['fid_gt'] = fid_gt

        return {**metrics}
    
    def update(self, recon_motion, gt_motion):
        rec_embeddings =self.embedding_model(recon_motion)
        gt_embeddings = self.embedding_model(gt_motion)

        self.recon_motion_embeddings.append(rec_embeddings)
        self.gt_motion_embeddings.append(gt_embeddings)