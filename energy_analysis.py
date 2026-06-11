"""
对抗点云能量差异分析脚本
===========================
用 GPointNet 的能量模型评估 HiT-ADV 生成的对抗点云与原点云的能量差异。

使用方法:
    python energy_analysis.py \
        --gpointnet_ckpt output/pytorch/chair_default_medium_*/checkpoint_2000.ckpt \
        --orig_npy data/chair_test.npy \
        --adv_npy adv_samples.npy \
        --num_point 2048
"""

import argparse
import torch
import numpy as np
from collections import defaultdict


def compute_energy(energy_net, point_cloud, batch_size=32, ref_sigma=0.3):
    """
    计算点云的完整能量: Eθ(X) = -fθ(X) + ||X||²/(2σ²)

    Args:
        energy_net: GPointNet 的能量网络
        point_cloud: [N, 3, num_points] 点云数组
        batch_size: 批处理大小
        ref_sigma: 参考分布标准差 (默认0.3)

    Returns:
        dict: {
            'energy': 总能量 Eθ (越低越自然),
            'score': 能量分数 fθ (越高越自然),
            'l2_penalty': L2 正则项 ||X||²/(2σ²)
        }
    """
    energy_net.eval()
    N = point_cloud.shape[0]

    all_scores = []
    all_l2 = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = torch.FloatTensor(point_cloud[i:i+batch_size]).cuda()
            # fθ(X): 能量分数
            score = energy_net(batch)  # [B, 1]
            all_scores.append(score.squeeze(-1).cpu())

            # ||X||²/(2σ²): L2 正则项
            l2_term = (batch ** 2).sum(dim=(1, 2)) / (2 * ref_sigma ** 2)
            all_l2.append(l2_term.cpu())

    scores = torch.cat(all_scores).numpy()
    l2_penalty = torch.cat(all_l2).numpy()
    energy = -scores + l2_penalty  # Eθ = -fθ + ||X||²/2σ²

    return {
        'energy': energy,
        'score': scores,
        'l2_penalty': l2_penalty
    }


def load_gpointnet_model(checkpoint_path, num_point=2048):
    """
    加载 GPointNet 预训练模型，只取 energy_net 部分。

    Args:
        checkpoint_path: .ckpt 文件路径
        num_point: 点云点数 (默认 2048)
    """
    import sys
    sys.path.insert(0, 'GPointNet/src')
    from model_point_torch import GPointNet
    import network_torch

    # 构造最小配置
    config = {
        'point_dim': 3,
        'num_point': num_point,
        'net_type': 'default_medium',
        'hidden_size': [[64, 64, 128, 256, 1024], [512, 256, 128, 64]],
        'swap_axis': True,
        'batch_norm': 'ln',
        'activation': 'ReLU',
        'ref_sigma': 0.3,
        'sample_step': 64,
        'step_size': 0.01,
        'noise_decay': 0,
        'langevin_decay': 0,
        'langevin_clip': 1,
        'activate_eval': 1,
    }

    model = GPointNet(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    model.eval()

    print(f"[✓] GPointNet loaded from {checkpoint_path}")
    print(f"    Energy net params: {sum(p.numel() for p in model.energy_net.parameters()):,}")
    return model.energy_net


def analyze_energy_diff(energy_net, orig_pc, adv_pc, ref_sigma=0.3):
    """
    计算原点云和对抗点云的能量差异，返回详细统计。

    Args:
        energy_net: GPointNet 能量网络
        orig_pc: 原点云 [N, 3, num_points]
        adv_pc: 对抗点云 [N, 3, num_points]
        ref_sigma: 参考分布标准差

    Returns:
        dict: 包含均值、中位数、分布等统计信息
    """
    print(f"\n{'='*60}")
    print(f"能量差异分析")
    print(f"{'='*60}")
    print(f"  原点云数量: {orig_pc.shape[0]}")
    print(f"  对抗点云数量: {adv_pc.shape[0]}")

    orig_energy = compute_energy(energy_net, orig_pc, ref_sigma=ref_sigma)
    adv_energy = compute_energy(energy_net, adv_pc, ref_sigma=ref_sigma)

    # 逐样本差异
    energy_diff = adv_energy['energy'] - orig_energy['energy']
    score_diff = orig_energy['score'] - adv_energy['score']  # orig score - adv score, 正=adv更差

    stats = {
        # 能量 (Eθ): 越低越自然
        'orig_energy_mean': float(np.mean(orig_energy['energy'])),
        'orig_energy_std': float(np.std(orig_energy['energy'])),
        'adv_energy_mean': float(np.mean(adv_energy['energy'])),
        'adv_energy_std': float(np.std(adv_energy['energy'])),
        'energy_diff_mean': float(np.mean(energy_diff)),
        'energy_diff_std': float(np.std(energy_diff)),
        # 分数 (fθ): 越高越自然
        'orig_score_mean': float(np.mean(orig_energy['score'])),
        'orig_score_std': float(np.std(orig_energy['score'])),
        'adv_score_mean': float(np.mean(adv_energy['score'])),
        'adv_score_std': float(np.std(adv_energy['score'])),
        'score_diff_mean': float(np.mean(score_diff)),
        'score_diff_std': float(np.std(score_diff)),
        # 统计检验
        'adv_energy_higher_pct': float(np.mean(energy_diff > 0) * 100),  # adv能量>原点的百分比
        'adv_score_lower_pct': float(np.mean(score_diff > 0) * 100),     # adv分数<原点的百分比
        # L2 正则项
        'orig_l2_mean': float(np.mean(orig_energy['l2_penalty'])),
        'adv_l2_mean': float(np.mean(adv_energy['l2_penalty'])),
    }

    # 打印结果
    print(f"\n{'─'*60}")
    print("  能量分数 fθ (越高越自然)")
    print(f"{'─'*60}")
    print(f"    原点云:   {stats['orig_score_mean']:.4f} ± {stats['orig_score_std']:.4f}")
    print(f"    对抗点云: {stats['adv_score_mean']:.4f} ± {stats['adv_score_std']:.4f}")
    print(f"    差异:     {stats['score_diff_mean']:.4f} ± {stats['score_diff_std']:.4f}")
    print(f"    对抗分数更低: {stats['adv_score_lower_pct']:.1f}%")

    print(f"\n{'─'*60}")
    print("  总能量 Eθ = -fθ + ||X||²/2σ² (越低越自然)")
    print(f"{'─'*60}")
    print(f"    原点云:   {stats['orig_energy_mean']:.4f} ± {stats['orig_energy_std']:.4f}")
    print(f"    对抗点云: {stats['adv_energy_mean']:.4f} ± {stats['adv_energy_std']:.4f}")
    print(f"    差异:     {stats['energy_diff_mean']:.4f} ± {stats['energy_diff_std']:.4f}")
    print(f"    对抗能量更高: {stats['adv_energy_higher_pct']:.1f}%")

    print(f"\n{'─'*60}")
    print("  L2 正则项 ||X||²/2σ²")
    print(f"{'─'*60}")
    print(f"    原点云:   {stats['orig_l2_mean']:.4f}")
    print(f"    对抗点云: {stats['adv_l2_mean']:.4f}")

    # 判断
    if stats['adv_energy_higher_pct'] > 80:
        print(f"\n  [结论] 对抗点云能量显著更高 ✓ — GPointNet 有效检测到对抗扰动")
    elif stats['adv_energy_higher_pct'] > 55:
        print(f"\n  [结论] 对抗点云能量略高 ~ — 有一定区分能力但不强")
    else:
        print(f"\n  [结论] 对抗点云能量无明显升高 ✗ — 对抗扰动逃逸了能量检测")

    return stats, energy_diff, score_diff


def per_class_analysis(energy_net, orig_pc, adv_pc, labels, ref_sigma=0.3):
    """
    按类别分析能量差异
    """
    print(f"\n{'='*60}")
    print("  按类别分析")
    print(f"{'='*60}")

    unique_labels = np.unique(labels)
    print(f"  {'类别':<8} {'数量':<8} {'Orig得分':<12} {'Adv得分':<12} {'得分差':<12}")
    print(f"  {'─'*52}")

    class_stats = {}
    for label in sorted(unique_labels):
        mask = labels == label
        if mask.sum() < 3:
            continue
        o = compute_energy(energy_net, orig_pc[mask], ref_sigma=ref_sigma)
        a = compute_energy(energy_net, adv_pc[mask], ref_sigma=ref_sigma)
        diff = float(np.mean(o['score']) - np.mean(a['score']))
        class_stats[label] = {
            'count': int(mask.sum()),
            'orig_score': float(np.mean(o['score'])),
            'adv_score': float(np.mean(a['score'])),
            'diff': diff
        }
        print(f"  {label:<8} {mask.sum():<8} {np.mean(o['score']):<12.4f} "
              f"{np.mean(a['score']):<12.4f} {diff:<12.4f}")

    return class_stats


def main():
    parser = argparse.ArgumentParser(description='对抗点云能量差异分析')
    parser.add_argument('--gpointnet_ckpt', type=str, required=True,
                        help='GPointNet checkpoint 路径')
    parser.add_argument('--orig_npy', type=str, required=True,
                        help='原点云 .npy 文件 [N, num_points, 3]')
    parser.add_argument('--adv_npy', type=str, required=True,
                        help='对抗点云 .npy 文件 [N, 3, num_points]')
    parser.add_argument('--labels', type=str, default=None,
                        help='标签 .npy 文件 [N] (可选, 用于按类别分析)')
    parser.add_argument('--num_point', type=int, default=2048,
                        help='点云点数 (默认 2048, HiT-ADV 用 1024 需上采样)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='计算批次大小')
    parser.add_argument('--ref_sigma', type=float, default=0.3,
                        help='GPointNet 参考分布 σ')
    parser.add_argument('--output', type=str, default='energy_diff_stats.npy',
                        help='输出统计文件')
    args = parser.parse_args()

    # 加载模型
    energy_net = load_gpointnet_model(args.gpointnet_ckpt, num_point=args.num_point)

    # 加载数据
    orig = np.load(args.orig_npy)  # [N, num_points, 3] or [N, 3, num_points]
    adv = np.load(args.adv_npy)

    # 统一格式为 [N, 3, num_points]
    if orig.shape[2] == 3:
        orig = orig.transpose(0, 2, 1)  # [N, 3, num_points]
    if adv.shape[2] == 3:
        adv = adv.transpose(0, 2, 1)

    # 如果点数不匹配，插值到一致
    if orig.shape[2] != args.num_point:
        print(f"[!] 原点云点数 {orig.shape[2]} ≠ {args.num_point}, 将上/下采样")
        # TODO: 添加 FPS 采样逻辑
        pass

    # 分析
    stats, energy_diff, score_diff = analyze_energy_diff(
        energy_net, orig, adv, ref_sigma=args.ref_sigma
    )

    # 按类别
    if args.labels:
        labels = np.load(args.labels)
        per_class_analysis(energy_net, orig, adv, labels, ref_sigma=args.ref_sigma)

    # 保存
    np.savez(args.output,
             stats=stats,
             energy_diff=energy_diff,
             score_diff=score_diff)
    print(f"\n[✓] 统计结果已保存到 {args.output}")


if __name__ == '__main__':
    main()
