import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync  # 轨迹时间戳同步
import evo.main_ape as main_ape  # 核心 APE 评估接口（与你的代码一致）
from evo.core.metrics import PoseRelation  # 误差类型（平移/旋转）
import os
import copy
import matplotlib.pyplot as plt
from evo.tools import plot  # 轨迹可视化（沿用你的导入逻辑）
from evo.tools.plot import PlotMode  # 可视化模式


def evaluate_two_tum_trajectories(
        gt_path,          # 参考轨迹（ground truth）路径（TUM格式）
        est_path,         # 待评估轨迹（估计结果）路径（TUM格式）
        save_dir="./",    # 结果保存目录
        plot_flag=False   # 是否可视化轨迹对比
):
    """
    评估两个 TUM 格式轨迹的绝对位姿误差（APE），包含 -a（姿态对齐）和 -s（尺度对齐）

    参数:
        gt_path: 参考轨迹文件路径（TUM格式：timestamp qx qy qz tx ty tz）
        est_path: 估计轨迹文件路径（TUM格式，与参考轨迹时间戳需重叠）
        save_dir: 评估结果（metrics.txt）保存目录
        plot_flag: 是否绘制轨迹对比图（3D XYZ 视图）

    返回:
        平移误差统计字典（max/mean/median/rmse 等）
    """
    # 1. 校验输入文件有效性
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"参考轨迹文件不存在：{gt_path}")
    if not os.path.exists(est_path):
        raise FileNotFoundError(f"估计轨迹文件不存在：{est_path}")
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在

    # 2. 读取 TUM 格式轨迹（与你的 EuRoC 代码读取逻辑一致）
    # 参考轨迹（ground truth）
    traj_ref = file_interface.read_tum_trajectory_file(gt_path)
    # 估计轨迹
    traj_est = file_interface.read_tum_trajectory_file(est_path)

    # traj_est.timestamps = traj_est.timestamps* 1e-18

    # 3. 时间戳同步（匹配两条轨迹的重叠时间范围，避免时间错位）
    # max_diff: 时间戳最大允许差值（秒），与你的代码中同步逻辑一致
    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        traj_ref, traj_est, max_diff=0.1
    )

    # 4. 执行 APE 评估（关键：-a 对应 align=True，-s 对应 correct_scale=True）
    # 平移误差评估（SLAM 主要关注指标）
    trans_result = main_ape.ape(
        traj_ref_sync,                # 参考轨迹（同步后）
        traj_est_sync,                # 估计轨迹（同步后）
        est_name="estimated_traj",    # 估计轨迹名称（用于结果显示）
        pose_relation=PoseRelation.translation_part,  # 评估平移误差
        align=True,                   # 对应 -a：对齐旋转+平移（SE(3)对齐）
        correct_scale=True            # 对应 -s：对齐尺度（Sim(3)对齐）
    )

    # （可选）旋转误差评估
    rot_result = main_ape.ape(
        traj_ref_sync,
        traj_est_sync,
        est_name="estimated_traj_rot",
        pose_relation=PoseRelation.rotation_part,  # 评估旋转误差（弧度）
        align=True,
        correct_scale=True
    )

    # 5. 保存评估结果到文本文件
    save_path = os.path.join(save_dir, "metrics_traj.txt")
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("TUM 轨迹评估结果（-a -s 对齐）\n")
        f.write("=" * 50 + "\n")
        f.write("【平移误差（单位：米）】\n")
        f.write(trans_result.pretty_str())  # 格式化输出平移误差统计
        f.write("\n" + "=" * 50 + "\n")
        f.write("【旋转误差（单位：弧度）】\n")
        f.write(rot_result.pretty_str())    # 格式化输出旋转误差统计

    print(f"✅ 评估完成！结果已保存至：{save_path}")

    # 6. 轨迹可视化（与你的 EuRoC 可视化逻辑一致）
    if plot_flag:
        # 复制并对齐估计轨迹（用于对比显示）
        traj_est_aligned = copy.deepcopy(traj_est_sync)
        traj_est_aligned.align(traj_ref_sync, correct_scale=True)  # 应用 Sim(3) 对齐

        # 绘制 3D 轨迹对比图（仅保留参考轨迹和已对齐轨迹）
        fig = plt.figure(figsize=(10, 8))
        # 轨迹字典：只包含参考轨迹和已对齐的估计轨迹
        traj_dict = {
            "参考轨迹 (Ground Truth)": traj_ref_sync,
            "估计轨迹 (已对齐 -a -s)": traj_est_aligned
        }
        # 绘制 XYZ 三维轨迹
        plot.trajectories(fig, traj_dict, PlotMode.xyz)
        plt.title("TUM Trajectory Comparison (Aligned with -a -s)")
        plt.tight_layout()
        plt.show()

    # 6. 返回平移误差统计（用于后续处理，如批量评估）
    return trans_result.stats


# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 1. 配置你的两个 TUM 轨迹文件路径
    GT_TUM_PATH = "/media/zhiyu/Seagate/Dataset/zed_exposure/s1/mav0/cam0/DROID-SLAM/gt_pose.txt"   # change to your path
    EST_TUM_PATH = "./results/zed/s1/CameraTrajectory_TUM.txt"  # change to your path
    SAVE_DIRECTORY = "./results/zed/s1"  # change to your path

    # 3. 执行评估（plot_flag=True 显示轨迹图）
    error_stats = evaluate_two_tum_trajectories(
        gt_path=GT_TUM_PATH,
        est_path=EST_TUM_PATH,
        save_dir=SAVE_DIRECTORY,
        plot_flag=True
    )

    # 4. 打印核心误差指标（在终端快速查看）
    print("\n📊 核心平移误差指标：")
    print(f"最大误差: {error_stats['max']:.6f} m")
    print(f"平均误差: {error_stats['mean']:.6f} m")
    print(f"均方根误差 (RMSE): {error_stats['rmse']:.6f} m")
