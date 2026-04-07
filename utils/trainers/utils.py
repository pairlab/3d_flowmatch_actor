def compute_metrics(pred, gt):
    # pred/gt are (B, L, 3+rot+1)
    pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
    # symmetric quaternion eval
    quat_l1 = (pred[..., 3:-1] - gt[..., 3:-1]).abs().sum(-1)
    quat_l1_ = (pred[..., 3:-1] + gt[..., 3:-1]).abs().sum(-1)
    select_mask = (quat_l1 < quat_l1_).float()
    quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
    # gripper openess — (B, T, nhand) to match pos_l2/quat_l1
    openess = ((pred[..., -1] >= 0.5) == (gt[..., -1] >= 0.5)).float()
    tr = 'traj_'

    # Trajectory metrics
    ret_1 = {
        tr + 'pos_l2': pos_l2.mean(),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
        tr + 'rot_l1': quat_l1.mean(),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
        tr + 'gripper': openess.mean()
    }
    ret_2 = {
        tr + 'pos_l2': pos_l2.mean(-1),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
        tr + 'rot_l1': quat_l1.mean(-1),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1),
        tr + 'gripper': openess.mean(-1)
    }

    # Per-arm metrics for bimanual (pred has arm dim: B, T, nhand, ...)
    if pred.ndim > 3:
        for arm in range(pred.shape[-2]):
            a = f'arm{arm}_'
            a_pos = pos_l2[..., arm]
            a_rot = quat_l1[..., arm]
            a_grip = openess[..., arm]
            ret_1[tr + a + 'pos_l2'] = a_pos.mean()
            ret_1[tr + a + 'pos_acc_001'] = (a_pos < 0.01).float().mean()
            ret_1[tr + a + 'rot_l1'] = a_rot.mean()
            ret_1[tr + a + 'rot_acc_0025'] = (a_rot < 0.025).float().mean()
            ret_1[tr + a + 'gripper'] = a_grip.mean()
            ret_2[tr + a + 'pos_l2'] = a_pos.mean(-1)
            ret_2[tr + a + 'pos_acc_001'] = (a_pos < 0.01).float().mean(-1)
            ret_2[tr + a + 'rot_l1'] = a_rot.mean(-1)
            ret_2[tr + a + 'rot_acc_0025'] = (a_rot < 0.025).float().mean(-1)
            ret_2[tr + a + 'gripper'] = a_grip.mean(-1)

    return ret_1, ret_2
