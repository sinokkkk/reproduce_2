"""
HiT-ADV 对抗生成 → GPointNet 能量分析 一键流水线
===================================================
运行 HiT-ADV 攻击生成对抗点云，同时保存原点云，
然后用 GPointNet 计算能量差异。

用法:
    python run_pipeline.py \
        --checkpoint HiT-ADV/Checkpoint/PN_NT.checkpoint \
        --gpointnet_ckpt gpointnet_ckpt.ckpt \
        --data_root ../../PC_Dataset \
        --output_dir ./results
"""

import os
import sys
import argparse
import numpy as np
import torch

# 添加 HiT-ADV 到 path
sys.path.insert(0, 'HiT-ADV')


def run_hitadv_attack(args):
    """
    运行 HiT-ADV 攻击，生成并保存对抗点云和原点云
    """
    print(f"\n{'='*60}")
    print("  Phase 1: HiT-ADV 对抗攻击")
    print(f"{'='*60}")

    from Dataset.ModelNet import ModelNetDataLoader
    from model.pointnet_cls import get_model
    from model import feature_models
    from ShapeAttack.HiT_ADV import HiT_ADV
    from util.adv_utils import UntargetedLogitsAdvLoss
    import FGM.CWPert_args

    # 加载数据
    data_path = os.path.join(args.data_root, 'modelnet40_normal_resampled')
    test_dataset = ModelNetDataLoader(
        root=data_path, args=args, split='test', process_data=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 加载 victim 模型
    model = feature_models.PointNetFeatureModel(
        args.num_class, normal_channel=False
    ).cuda()
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    print(f"[✓] Victim model loaded: {args.checkpoint}")

    # 初始化 attacker
    adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
    CWPerturb_args = FGM.CWPert_args.get_args()
    attacker = HiT_ADV(
        model, adv_func=adv_func,
        attack_lr=CWPerturb_args.attack_lr,
        central_num=args.central_num,
        total_central_num=args.total_central_num,
        init_weight=10., max_weight=80.,
        binary_step=CWPerturb_args.binary_step,
        num_iter=CWPerturb_args.num_iter,
        clip_func=None,
        cd_weight=args.cd_weight,
        ker_weight=args.ker_weight,
        hide_weight=args.hide_weight,
        curv_loss_knn=args.curv_loss_knn,
        max_sigm=args.max_sigm,
        min_sigm=args.min_sigm,
        budget=args.budget
    )

    # 运行攻击
    all_orig = []
    all_adv = []
    all_labels = []
    total_success = 0
    total_samples = 0

    os.makedirs(args.output_dir, exist_ok=True)

    for batch_idx, (data, label) in enumerate(test_loader):
        if batch_idx >= args.max_batches:
            break

        print(f"\n--- Batch {batch_idx+1}/{args.max_batches} ---")

        # 提取 xyz (前3通道)
        xyz_orig = data[:, :, :3].cuda().transpose(1, 2).contiguous()  # [B, 3, N]
        label = label.cuda()

        # 生成对抗点云
        adv_data, success_num = attacker.attack(data, label)

        # 收集结果
        all_orig.append(xyz_orig.cpu().numpy())           # [B, 3, N]
        all_adv.append(adv_data)                            # [B, 3, N]
        all_labels.append(label.cpu().numpy())

        total_success += success_num
        total_samples += data.shape[0]

    # 合并
    orig_pc = np.concatenate(all_orig, axis=0)
    adv_pc = np.concatenate(all_adv, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"\n[✓] 攻击完成: {total_success}/{total_samples} 成功 "
          f"({100*total_success/total_samples:.1f}%)")

    # 保存
    np.save(os.path.join(args.output_dir, 'orig_pc.npy'), orig_pc)
    np.save(os.path.join(args.output_dir, 'adv_pc.npy'), adv_pc)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels)
    print(f"[✓] 保存: orig_pc.npy ({orig_pc.shape}), adv_pc.npy ({adv_pc.shape})")

    return orig_pc, adv_pc, labels


def run_energy_analysis(args, orig_pc, adv_pc, labels):
    """
    用 GPointNet 计算能量差异
    """
    print(f"\n{'='*60}")
    print("  Phase 2: GPointNet 能量分析")
    print(f"{'='*60}")

    sys.path.insert(0, 'GPointNet/src')

    # 使用独立脚本中的函数
    from energy_analysis import load_gpointnet_model, analyze_energy_diff, per_class_analysis

    energy_net = load_gpointnet_model(
        args.gpointnet_ckpt,
        num_point=orig_pc.shape[2]
    )

    # 统一点数 (GPointNet 默认 2048, HiT-ADV 是 1024)
    target_n = energy_net.local[0].in_channels if hasattr(energy_net.local[0], 'in_channels') else orig_pc.shape[2]
    # 实际上 GPointNet 读取的是 point_dim=3, 所以是 [B, 3, N], N 可以是任意值
    # 直接传入即可

    # 分析
    stats, energy_diff, score_diff = analyze_energy_diff(
        energy_net, orig_pc, adv_pc, ref_sigma=0.3
    )

    # 按类别
    per_class_analysis(energy_net, orig_pc, adv_pc, labels, ref_sigma=0.3)

    # 保存详细结果
    np.savez(
        os.path.join(args.output_dir, 'energy_analysis.npz'),
        stats=stats,
        energy_diff=energy_diff,
        score_diff=score_diff,
        labels=labels
    )
    print(f"\n[✓] 全部分析结果保存到 {args.output_dir}/")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='HiT-ADV + GPointNet 对抗能量分析流水线'
    )

    # HiT-ADV 参数
    parser.add_argument('--checkpoint', type=str,
                        default='HiT-ADV/Checkpoint/PN_NT.checkpoint',
                        help='HiT-ADV victim model checkpoint')
    parser.add_argument('--data_root', type=str,
                        default='../../PC_Dataset',
                        help='ModelNet40 数据目录')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='攻击批次大小')
    parser.add_argument('--num_class', type=int, default=40)
    parser.add_argument('--num_point', type=int, default=1024,
                        help='HiT-ADV 点数')
    parser.add_argument('--budget', type=float, default=0.55,
                        help='攻击预算')
    parser.add_argument('--kappa', type=float, default=30.)
    parser.add_argument('--central_num', type=int, default=192)
    parser.add_argument('--total_central_num', type=int, default=256)
    parser.add_argument('--cd_weight', type=float, default=0.0001)
    parser.add_argument('--ker_weight', type=float, default=1.)
    parser.add_argument('--hide_weight', type=float, default=1.)
    parser.add_argument('--curv_loss_knn', type=int, default=16)
    parser.add_argument('--max_sigm', type=float, default=1.2)
    parser.add_argument('--min_sigm', type=float, default=0.1)
    parser.add_argument('--max_batches', type=int, default=5,
                        help='最多运行多少 batch (测试用)')

    # GPointNet 参数
    parser.add_argument('--gpointnet_ckpt', type=str, required=True,
                        help='GPointNet checkpoint 路径')

    # 输出
    parser.add_argument('--output_dir', type=str, default='./energy_results',
                        help='结果输出目录')

    args = parser.parse_args()

    # Phase 1: 生成对抗点云
    orig_pc, adv_pc, labels = run_hitadv_attack(args)

    # Phase 2: 能量分析
    if args.gpointnet_ckpt:
        run_energy_analysis(args, orig_pc, adv_pc, labels)
    else:
        print("[!] 未提供 --gpointnet_ckpt，跳过能量分析")
        print(f"[✓] 对抗点云已保存到 {args.output_dir}/")


if __name__ == '__main__':
    main()
