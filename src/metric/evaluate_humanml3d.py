import os
from pathlib import Path
import numpy as np
from copy import deepcopy
import torch
import sys
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.inference.inference_util import load_config
from src.models.pl_module.cross_energy_edit_dance_joint import CrossEnergyEditDanceJoint
from src.dataloader.dataset_loader import get_datasets
from src.metric.evaluator_wrapper import Evaluators
from src.metric.back_process import back_process
from src.metric.humanml_metrics import *

def evaluation(data_loader, model, ep, best_fid, clip_score_old, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, device, cal_mm=True):
    eval_mean = np.load('./src/metric/eval_mean.npy', allow_pickle=True)
    eval_std = np.load('./src/metric/eval_std.npy', allow_pickle=True)
    motion_pred_list = []
    motion_multimodality = []
    motion_annotation_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    clip_score_real = 0
    clip_score_gt = 0
    skate_ratio_sum = 0
    dist_sum = 0
    traj_err = []

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    model.to(device)
    model.eval()
    for i, batch in enumerate(tqdm(data_loader)):
        bs = batch['motion'].shape[0]
        pos_one_hots = batch['pos_one_hots']
        sent_len = batch['sent_len']
        model_input = {
            'audio': batch['audio'].to(device),
            'attributes': batch['attributes'].to(device),
            'description': batch['description'].to(device),
            'label_index': batch['label_index'].to(device),
            'audio_mask': batch['audio_mask'].to(device),
            'att_mask': batch['att_mask'].to(device),
        }
        if i < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30)):
                pred = model.forward(model_input)
                pred_pose = pred['motion']
                pred_pose_joints = pred['motion_fk']

                pred_motions = []
                for j in range(pred_pose.shape[0]):
                    pred_motions.append(back_process(pred_pose[j], pred_pose_joints[j]))

                pred_motions = np.stack(pred_motions, axis=0)
                pred_motions = (pred_motions - eval_mean) / eval_std
                pred_motions = torch.from_numpy(pred_motions).to(device)
                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(
                    batch['word_embeddings'],
                    pos_one_hots,
                    sent_len,
                    batch['text'],
                    pred_motions,
                    m_lens=150
                )
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred
        else:
            pred = model.forward(model_input)
            pred_pose = pred['motion']
            pred_pose_joints = pred['motion_fk']
            pred_motions = []
            for j in range(pred_pose.shape[0]):
                pred_motions.append(back_process(pred_pose[j], pred_pose_joints[j]))
            pred_motions = np.stack(pred_motions, axis=0)
            pred_motions = (pred_motions - eval_mean) / eval_std
            pred_motions = torch.from_numpy(pred_motions).to(device)
            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(
                    batch['word_embeddings'],
                    pos_one_hots,
                    sent_len,
                    batch['text'],
                    pred_motions,
                    m_lens=150
                )
            batch_clip_score_pred = 0
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        
        # Compute for GT
        pose = batch['motion']
        pose_joint = batch['motion_joint']
        gt_motions = []
        for j in range(pose.shape[0]):
            gt_motions.append(back_process(pose[j], pose_joint[j]))
        gt_motions = np.stack(gt_motions, axis=0)
        gt_motions = (gt_motions - eval_mean) / eval_std
        gt_motions = torch.from_numpy(gt_motions).to(device)
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(
                batch['word_embeddings'],
                pos_one_hots,
                sent_len,
                batch['text'],
                gt_motions,
                m_lens=150
            )
        batch_clip_score = 0
        for j in range(bs):
            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score
        clip_score_gt += batch_clip_score
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs


    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (f"--> \t Eva. Ep/Re {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity."
           f" {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision},"
           f" matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
           f" multimodality. {multimodality:.4f}, clip score. {clip_score_real}")
    print(msg)



    if fid < best_fid:
        best_fid, best_ep = fid, ep
        save=True

    if matching_score_pred < best_matching:
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        best_div = diversity

    if R_precision[0] > best_top1:
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        best_top3 = R_precision[2]

    if clip_score_real > clip_score_old:
        clip_score_old = clip_score_real

    if not cal_mm:
        multimodality = 0
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, clip_score_old, save


def main(data_from='test', id=None):
    config, args = load_config()
    config.evaluation = True
    config.data.dataset=['HumanML3D']
    exp_name = config.name
    root_dir = os.path.join('./results', exp_name, 'HumanML3D')
    os.makedirs(root_dir, exist_ok=True)

    save_path = os.path.join(root_dir, 'motions')
    os.makedirs(save_path, exist_ok=True)


    # 2. Load data for evaluation: train, val, test
    if id is not None:
        data_from = 'sample'
    dataset = get_datasets(config, data_from, id)
    dataset.setup(stage=data_from, id=id)
    dataset_info = dataset.get_dataset_info()
    data_loader = dataset.test_dataloader(shuffle=False)
    
    if config.model.name == 'vae':
        model = EditDance(config, dataset_info, mode='inference')
    elif config.model.name == 'cross_latent_diffusion':
        model = CrossEnergyEditDance(config, dataset_info, mode='inference')
    elif 'joint' in config.model.name:
        model = CrossEnergyEditDanceJoint(config, dataset_info, mode='inference')
    else:
        raise ValueError(f"Model {config.model.name} not supported")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mm = []
    clip_scores = []

    repeat_time = 2

    eval_wrapper = Evaluators(device)
    f = open(os.path.join(root_dir, 'metrics.txt'), 'w')

    for i in range(repeat_time):
        with torch.no_grad():
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mm, clip_score = 1000, 0, 0, 0, 0, 100, 0, -1
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mm, clip_score, _ = evaluation(
                data_loader, model, i, best_fid=best_fid, clip_score_old=clip_score,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper, device=device, cal_mm=True)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        mm.append(best_mm)
        clip_scores.append(clip_score)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mm = np.array(mm)
    clip_scores = np.array(clip_scores)

    print(f'final result:')
    print(f'final result:', file=f, flush=True)

    msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n" \
                f"\tCLIP-Score:{np.mean(clip_scores):.3f}, conf.{np.std(clip_scores) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
    print(msg_final)
    print(msg_final, file=f, flush=True)
    f.close()


if __name__ == '__main__':
    main()