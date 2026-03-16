import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cfr_curves(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return

    # 读取数据
    try:
        # 读取文件，跳过标题行（原文件标题为 "lnx color_r color_g color_b"）
        data = np.loadtxt(file_path, skiprows=1)

        # 提取数据列：横坐标为原 ln(x)（对应 log-Exposure），纵坐标为原始 [0,1] 范围的颜色值
        log_exposure = data[:, 0]  # 横坐标：log-Exposure（原代码中的 lnx）
        color_r_01 = data[:, 1]    # 原始红通道值 [0,1]
        color_g_01 = data[:, 2]    # 原始绿通道值 [0,1]
        color_b_01 = data[:, 3]    # 原始蓝通道值 [0,1]

        # 核心修改：将 [0,1] 范围的颜色值归一化到 [0,255]（Image Intensity 常用范围）
        # 公式：目标值 = 原始值 * 255（若原始值已严格在 [0,1] 内，无需额外裁剪；若有溢出则用 clip 限制范围）
        color_r_255 = (color_r_01 * 255).clip(0, 255)  # 避免极端值溢出 [0,255]
        color_g_255 = (color_g_01 * 255).clip(0, 255)
        color_b_255 = (color_b_01 * 255).clip(0, 255)

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 创建图形（保持原尺寸 10x6，兼顾清晰度和显示效果）
    plt.figure(figsize=(10, 6))

    # 绘制三条曲线（颜色与通道对应，线型不变，图例标注调整为包含范围说明）
    plt.plot(log_exposure, color_r_255, 'r-', linewidth=1.5, label='Red Channel (0-255)')
    plt.plot(log_exposure, color_g_255, 'g-', linewidth=1.5, label='Green Channel (0-255)')
    plt.plot(log_exposure, color_b_255, 'b-', linewidth=1.5, label='Blue Channel (0-255)')

    # 关键修改：横纵坐标标签与需求一致
    plt.title('CFR Curves (log-Exposure vs Image Intensity)', fontsize=14, pad=15)  # 标题补充说明
    plt.xlabel('log-Exposure', fontsize=12, labelpad=10)  # 横坐标：log-Exposure
    plt.ylabel('Image Intensity', fontsize=12, labelpad=10)  # 纵坐标：Image Intensity

    # 优化：固定纵坐标范围为 [0,255]，确保曲线显示更直观（避免自动缩放导致范围偏移）
    plt.ylim(0, 255)
    # 可选：添加纵坐标刻度（每 50 一个刻度，符合 0-255 范围习惯）
    plt.yticks(np.arange(0, 256, 50))

    # 保留网格和图例（网格线透明度调整为 0.5，避免遮挡曲线）
    plt.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    plt.legend(fontsize=11, loc='best')  # loc='best' 自动选择最优图例位置

    # 调整布局，避免标签被截断
    plt.tight_layout()

    # 显示图形
    plt.show()

# 使用示例（路径保持与您原代码一致）
if __name__ == "__main__":
    cfr_file_path = "../results/euroc_dark/V1_02_medium/cfr_curve.txt"   # revise the sequence name
    plot_cfr_curves(cfr_file_path)