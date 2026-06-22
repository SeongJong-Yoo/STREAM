import torch
import torch.nn.functional as F

# =============================================================================
# Label Coverage Weighting for Partial Label Handling (Option 1 - Fixed Window)
# =============================================================================

def compute_label_coverage_weights(label_index, start_threshold=0.1, min_weight=0.3):
    """
    Compute per-frame weights based on label segment completeness.

    For fixed-window slicing, some label segments may be incomplete (e.g., only
    capturing the end of a motion phrase). This function down-weights frames from
    incomplete label segments to reduce conflicting supervision.

    Args:
        label_index: [B, T] - normalized progress values (0-1) that reset at label boundaries
                     e.g., [0.0, 0.1, ..., 1.0, 0.0, 0.1, ..., 1.0] for two consecutive labels
                     0.0 = start of label, 1.0 = end of label
        start_threshold: if a segment starts above this value, consider incomplete (default 0.1)
        min_weight: minimum weight for very incomplete segments (default 0.3)

    Returns:
        weights: [B, T] - per-frame weights in [min_weight, 1.0]
                 Returns None if label_index is None
    """
    if label_index is None:
        return None

    B, T = label_index.shape
    device = label_index.device
    weights = torch.ones(B, T, device=device, dtype=torch.float32)

    for b in range(B):
        idx = label_index[b]

        # Find label boundaries (where index decreases = new label starts)
        boundaries = [0]
        for t in range(1, T):
            if idx[t] < idx[t-1]:  # Reset detected (new label)
                boundaries.append(t)
        boundaries.append(T)

        # For each label segment, compute weight based on completeness
        for i in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[i], boundaries[i+1]
            start_value = idx[seg_start].item()  # 0-1 normalized start position
            end_value = idx[seg_end - 1].item()  # 0-1 normalized end position

            # Calculate coverage: what fraction of the label we captured
            # If start=0.0, end=1.0 → coverage=1.0 (complete)
            # If start=0.5, end=1.0 → coverage=0.5 (captured second half)
            # If start=0.0, end=0.3 → coverage=0.3 (captured first 30%)
            coverage = end_value - start_value

            # Handle edge case where end_value might equal start_value (single frame)
            coverage = max(coverage, 0.01)

            if start_value <= start_threshold:
                # Complete or nearly complete start
                # Weight based on how much we captured (penalize early cutoff too)
                weight = min_weight + (1.0 - min_weight) * min(coverage / (1.0 - start_value + 1e-6), 1.0)
            else:
                # Incomplete start: entered label partway through
                # Weight based on coverage
                weight = min_weight + (1.0 - min_weight) * coverage

            weights[b, seg_start:seg_end] = weight

    return weights


def apply_coverage_weights_to_mask(mask, coverage_weights):
    """
    Combine binary mask with coverage weights.

    Args:
        mask: [B, T] - binary mask (1 for valid, 0 for padding)
        coverage_weights: [B, T] - coverage-based weights from compute_label_coverage_weights

    Returns:
        combined_weights: [B, T] - mask * coverage_weights
    """
    if coverage_weights is None:
        return mask
    if mask is None:
        return coverage_weights

    # Ensure same dtype
    mask_float = mask.float() if mask.dtype == torch.bool else mask
    return mask_float * coverage_weights


metric_monitor = {
    "APE root": "Metrics/APE_root",
    "APE mean pose": "Metrics/APE_mean_pose",
    "AVE root": "Metrics/AVE_root",
    "AVE mean pose": "Metrics/AVE_mean_pose",
    "R_TOP_1": "Metrics/R_precision_top_1",
    "R_TOP_2": "Metrics/R_precision_top_2",
    "R_TOP_3": "Metrics/R_precision_top_3",
    "gt_R_TOP_1": "Metrics/gt_R_precision_top_1",
    "gt_R_TOP_2": "Metrics/gt_R_precision_top_2",
    "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
    "FID": "Metrics/FID",
    "gt_FID": "Metrics/gt_FID",
    "Diversity": "Metrics/Diversity",
    "gt_Diversity": "Metrics/gt_Diversity",
    "MM dist": "Metrics/Matching_score",
    "Accuracy": "Metrics/accuracy",
    "gt_Accuracy": "Metrics/gt_accuracy",
}

FOOT_JOINT_INDEX = {
    'motorica': [15, 16],
    'smpl': [7, 8, 10, 11]
    # 'smpl': [10, 11]
}

def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def contrastive_divergence(positive_energy, negative_energy):
    loss = 0
    # loss = F.relu(positive_energy.mean() - negative_energy.mean() + 1e-5)
    # loss = positive_energy.mean() - negative_energy.mean() # + 0.01 * (positive_energy**2 + negative_energy**2).mean()
    # print(f"positive_energy: {positive_energy.mean().item():.3f}", f"negative_energy: {negative_energy.mean().item():.3f}", f"Diff: {(positive_energy.mean() - negative_energy.mean()).item():.3f}")
    if positive_energy.ndim == 2:
        B, T = positive_energy.shape
    elif positive_energy.ndim == 1:
        B = positive_energy.shape[0]
        T = 1
    else:
        raise ValueError(f"Invalid shape: {positive_energy.shape}")
    positive_energy = positive_energy.reshape(B*T, -1)
    negative_energy = negative_energy.reshape(B*T, -1)
    labels = torch.zeros(B*T, device=positive_energy.device).long()
    total_energy = torch.cat([positive_energy, negative_energy], dim=-1)
    loss += F.cross_entropy(-1 * total_energy, labels)
    return loss

def regularization(value1, value2=None, sqrt=False):
    if sqrt:
        loss = torch.mean(torch.sqrt(value1 ** 2))
        if value2 is not None:
            loss += torch.mean(torch.sqrt(value2 ** 2))
    else:
        loss = torch.mean(value1 ** 2)
        if value2 is not None:
            loss += torch.mean(value2 ** 2)
    return loss.mean()

def MSELoss(a, b, weights=None, mask=None):
    mse = torch.nn.MSELoss(reduction='none')(a, b)
    mse = mse.mean(dim=-1)
    if mask is not None:
        total_sum = sum_flat(mse * mask)
        total_count = sum_flat(mask)
        mse = total_sum / total_count
        return torch.mean(mse)

    if weights is None:
        return torch.mean(mse)
    else:
        while mse.ndim != 1:
            mse = mse.mean(dim=-1)
        loss = weights * mse
        return torch.mean(loss)

def L2_loss(a, b, weights=None, mask=None):
    # return torch.mean(torch.norm(a - b, dim=-1, p=2))
    # # L2 = torch.norm(a - b, dim=-1, p=2)
    L2 = torch.norm(a - b, dim=-1, p=2)
    if L2.ndim == 3:
        L2 = L2.mean(dim=-1)
    if mask is not None:
        total_sum = sum_flat(L2 * mask)
        total_count = sum_flat(mask)
        L2 = total_sum / total_count
        return torch.mean(L2)

    if weights is None:
        return torch.mean(L2)
    else:
        while L2.ndim != 1:
            if L2.shape[-1] == mask.shape[-1]:
                L2 = L2 * mask
            L2 = L2.mean(dim=-1)
        loss = weights * L2
        return torch.mean(loss)

def L1_loss(a, b, mask=None):
    if mask is not None:
        l1 = torch.norm(a - b, dim=-1, p=1)
        while l1.ndim > 2:
            l1 = l1.mean(dim=-1)
        l1 = l1 * mask
        total_sum = sum_flat(l1)
        total_count = sum_flat(mask)
        l1 = total_sum / total_count
        return torch.mean(l1)

    return torch.mean(torch.norm(a - b, dim=-1, p=1))

def compute_vel(a):
    if a.ndim == 3:
        a_vel = a[1:] - a[:-1]
        return torch.concat((a_vel, torch.zeros_like(a_vel[-1:])), dim=0)
    elif a.ndim == 4:
        a_vel = a[:, 1:] - a[:, :-1]
        return torch.concat((a_vel, torch.zeros_like(a_vel[:, -1:])), dim=1)
    else:
        raise ValueError(f"Invalid shape: {a.shape}")

def compute_acc(a):
    a_acc = a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]
    return torch.concat((a_acc, torch.zeros_like(a_acc[:, -2:])), dim=1)

# def mpjve(a, b):
#     """
#     Compute similarity loss between two pose velocities
#     Input: batch, frame, joint, dim
#     """
#     a_vel = a[:, 1:] - a[:, :-1]
#     b_vel = b[:, 1:] - b[:, :-1]

#     return torch.mean(F.normalize(a_vel - b_vel, dim=-1, p=2, eps=1e-6))

# def mpjae(a, b):
#     """
#     Compute similarity loss between two pose accelerations
#     Input: batch, frame, joint, dim
#     """
#     a_acc = a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]
#     b_acc = b[:, 2:] - 2 * b[:, 1:-1] + b[:, :-2]

#     return torch.mean(F.normalize(a_acc - b_acc, dim=-1, p=2, eps=1e-6))

def cos_sim(a, b):
    """ 
    Compute cosine similarity
    Input: batch, frame, joint, dim
    """
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    sim = torch.einsum('bfjd,bfjd->bfj', a_norm, b_norm)
 
    return torch.mean(1 - sim)

def max_cos_sim(a, b):
    """ 
    Compute max cosine similarity
    Input: batch, frame, joint, dim
    """
    m = torch.maximum(torch.norm(a, dim=-1), torch.norm(b, dim=-1))
    if len(a.shape) == 4:
        sim = torch.einsum('bfjd,bfjd->bfj', a, b) / m**2
    else:
        sim = torch.einsum('bfj,bfj->bf', a, b) / m**2
 
    return torch.mean(1 - sim)


def weighted_cos_sim(ref, pred, mask=None):
    """
    Compute weighted (based on joint-axis) cosine similarity
    Input: 
        batch, frame, joint, dim
    """
    weight = torch.norm(ref, dim=-1)
    weight = F.normalize(weight, dim=-1)
    a_norm = F.normalize(pred, dim=-1)
    b_norm = F.normalize(ref, dim=-1)
    if len(a_norm.shape) == 4:
        sim = torch.einsum('bfjd, bfjd -> bfj', a_norm, b_norm)
    elif len(a_norm.shape) == 3:
        sim = torch.einsum('bfj, bfj -> bf', a_norm, b_norm)
    else:
        raise ValueError(f"Sim computation failed: Invalid shape: {a_norm.shape}")
    # sim = weight * sim
    if mask is not None:
        if sim.ndim == 3:
            sim = sim.mean(dim=-1)
        total_sum = sum_flat(sim * mask)
        total_count = sum_flat(mask)
        sim = total_sum / total_count
        return 1 - torch.mean(sim) + 0.5 * L1_loss(pred, ref, mask)

    return 1 - torch.mean(sim)


def foot_skate_loss(a, gt, data_rep='smpl', threshold=0.05):
    """
    Compute foot skate loss
    Input:
        batch, frame, joint, dim
    Output:
        loss
    """
    foot_idx = FOOT_JOINT_INDEX['smpl']
    foot = a[:, :, foot_idx]
    foot_min = torch.stack([torch.min(foot[i, :, :, 1]) for i in range(foot.shape[0])])
    while foot_min.ndim != foot.ndim:
        foot_min = foot_min.unsqueeze(1)
    foot = foot - foot_min
    mask = torch.where(torch.abs(foot[:, :, :, 1]) < threshold, 1, 0) # y axis is the height
    vel = compute_vel(foot)
    vel = torch.stack((vel[..., 0], vel[..., 2]), dim=-1)
    pred_skating_loss = torch.mean(mask * torch.norm(vel, dim=-1, p=2))
    # value = compute_acc(foot)

    # Compute foot contact loss
    foot_gt = gt[:, :, foot_idx]
    foot_gt_min = torch.stack([torch.min(foot_gt[i, :, :, 1]) for i in range(foot_gt.shape[0])])
    while foot_gt_min.ndim != foot_gt.ndim:
        foot_gt_min = foot_gt_min.unsqueeze(1)
    foot_gt = foot_gt - foot_gt_min
    mask_gt = torch.where(torch.abs(foot_gt[:, :, :, 1]) < threshold,  1, 0)
    # diff = torch.norm(foot - foot_gt, dim=-1, p=2)
    diff = torch.abs(foot[..., 1] - foot_gt[..., 1])
    foot_contact_loss = (torch.mean(mask_gt * diff) + torch.mean(mask * diff))
    return pred_skating_loss + 0.5 * foot_contact_loss


class InfoNCELoss():
    def __init__(self, temperature):
        self.temperature = temperature

    def music_id(self, key, dataset='aist'):
        #TODO: Implement compatible music id
        if dataset == 'aist':
            id = key.split('_')[4]
            return id
        elif dataset == 'motorica':
            id = key.split('_')[5]
            return id
        else:
            raise ValueError(f"Dataset {dataset} not supported")

    def similarity_matrix(self, a, b):
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)

        return a_norm @ b_norm.T / self.temperature
    
    def compute_infonce(self, sim_matrix, label, keys, dataset, mask_keys=True):
        if mask_keys:
            keys_tensor = torch.tensor([hash(self.music_id(key, dataset)) for key in keys], device=sim_matrix.device)
            id_matrix = keys_tensor.unsqueeze(1) == keys_tensor.unsqueeze(0)
            diagonal_mask = ~torch.eye(len(keys), dtype=torch.bool, device=sim_matrix.device)
            sim_matrix = torch.where(id_matrix & diagonal_mask, -torch.inf, sim_matrix)

        loss = F.cross_entropy(sim_matrix, label) + F.cross_entropy(sim_matrix.T, label)

        return loss / 2

    
    def __call__(self, a, b, keys, data_name, mask, mask_keys=True):
        a = a[mask]
        b = b[mask]
        batch_size = a.shape[0]
        # sim_matrix = self.similarity_matrix(a, b)
        label = torch.arange(batch_size, device=a.device)

        total_loss = 0
        if len(a.shape) == 3:
            # Compute similarity matrix based on time axis
            for i in range(a.shape[1]):
                a_i = a[:, i, :]
                b_i = b[:, i, :]
                sim_matrix_i = self.similarity_matrix(a_i, b_i)
                loss_i = self.compute_infonce(sim_matrix_i, label, keys, data_name, mask_keys)
                total_loss += loss_i
            total_loss /= a.shape[1]
        else:
            sim_matrix = self.similarity_matrix(a, b)
            total_loss = self.compute_infonce(sim_matrix, label, keys, data_name, mask_keys)

        return total_loss
                

        # if mask_keys:
        #     keys_tensor = torch.tensor([hash(self.music_id(key, dataset)) for key in keys], device=device)
        #     id_matrix = keys_tensor.unsqueeze(1) == keys_tensor.unsqueeze(0)
        #     diagonal_mask = ~torch.eye(len(keys), dtype=torch.bool, device=device)
        #     sim_matrix = torch.where(id_matrix & diagonal_mask, -torch.inf, sim_matrix)

        # loss = F.cross_entropy(sim_matrix, label) + F.cross_entropy(sim_matrix.T, label)
        # return loss / 2

class DisentangleLoss():
    """
    Disentangle loss to push music and text embeddings into orthogonal subspaces.
    Uses global subspace separation (all music orthogonal to all text).
    """
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def similarity_matrix(self, a, b, diagonal_only=False):
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)

        matrix = a_norm @ b_norm.T / self.temperature
        if diagonal_only:
            matrix = torch.diag(matrix)
        return torch.abs(matrix)

    def __call__(self, music, text, mask=None):
        """
        Args:
            music: [B, T, D] music embeddings (frame-level)
            text: [B, 2, D] text embeddings (attributes + description)
            mask: [B, T] boolean mask for valid frames
        """
        # Pool music over time with mask
        if mask is not None:
            mask_float = mask.float()
            music = (music * mask_float.unsqueeze(-1)).sum(1) / mask_float.sum(1).clamp(min=1).unsqueeze(-1)
        else:
            music = music.mean(dim=1)  # [B, D]

        # Pool text over attributes/description
        text = text.mean(dim=1)  # [B, D]

        if music.shape != text.shape:
            raise ValueError(f"music and text must have the same shape: {music.shape} != {text.shape}")

        # Global subspace separation: any music should be orthogonal to any text
        return torch.mean(self.similarity_matrix(music, text, diagonal_only=False))


class MotionTextContrastiveLoss():
    """
    Contrastive loss to align motion and text embeddings.
    Uses InfoNCE-style loss to pull matching pairs together.
    """
    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def __call__(self, motion_embed, text_embed):
        """
        Args:
            motion_embed: [B, D] motion embeddings (pooled, detached)
            text_embed: [B, D] text embeddings (pooled)
        """
        # Normalize embeddings
        motion_norm = F.normalize(motion_embed, dim=-1)
        text_norm = F.normalize(text_embed, dim=-1)

        # Compute similarity matrix
        logits = motion_norm @ text_norm.T / self.temperature  # [B, B]

        # Labels: diagonal elements are positive pairs
        batch_size = motion_embed.shape[0]
        labels = torch.arange(batch_size, device=motion_embed.device)

        # Symmetric loss
        loss_m2t = F.cross_entropy(logits, labels)
        loss_t2m = F.cross_entropy(logits.T, labels)

        return (loss_m2t + loss_t2m) / 2