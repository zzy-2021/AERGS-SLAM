import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation


def associate_frames(tstamp_image, tstamp_pose, max_dt=0.08):
    """ pair images, depths, and poses """
    associations = []
    for i, t in enumerate(tstamp_image):
        j = np.argmin(np.abs(tstamp_pose - t))
        if (np.abs(tstamp_pose[j] - t) < max_dt):
            associations.append((i, j))
    return associations

def loadPose(path):
    poses = []
    """ 
    with open(path, "r") as fin:
        lines = fin.readlines()
    for line in lines:
        line = np.array(list(map(float, line.split())))
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(line[4:]).as_matrix()
        pose[:3, 3] = line[1:4]
        poses.append(pose)
    """
    pose_data = np.loadtxt(path, delimiter=' ', dtype=np.unicode_)
    pose_vecs = pose_data[:, 1:].astype(np.float32)
    tstamp = pose_data[:, 0].astype(np.float64)

    for pose_vec in pose_vecs:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pose_vec[3:]).as_matrix()
        pose[:3, 3] = pose_vec[:3]
        poses.append(pose)
        #print(pose);exit()
    return poses, tstamp

def loadEuRoC(path):
    #print(path)
    color_paths = sorted(glob.glob(os.path.join(path, 'mav0/cam0/data/*.png')))
    #print(color_paths)
    tstamp = [np.float64(x.split('/')[-1][:-4])/1e9 for x in color_paths]
    return color_paths, tstamp



import os
import shutil
import tempfile

def convert_tum_timestamp_to_seconds(pose_path, backup_original=False):
    """
    转换TUM轨迹文件中第一列的纳秒时间戳为秒（添加小数点），直接覆盖源文件
    逻辑：先判断第一列是否为纳秒时间戳，仅对纳秒格式数据进行转换，默认备份原文件

    参数:
        pose_path (str): TUM轨迹文件路径（CameraTrajectory_TUM.txt）
        backup_original (bool): 是否备份原文件（默认True，覆盖前先备份，避免数据丢失）

    返回:
        str: 处理后的源文件路径（即输入的pose_path）

    异常:
        FileNotFoundError: 输入文件不存在
        IsADirectoryError: 输入路径是文件夹而非文件
        ValueError: 文件内容格式异常（无有效行或第一列非数字）
        IOError: 文件读写过程中出现错误（如权限不足）
    """
    # -------------------------- 纳秒时间戳判断函数（保留原逻辑） --------------------------
    def is_nanosecond_timestamp(timestamp_str):
        """判断字符串是否为TUM数据集格式的纳秒时间戳"""
        # 条件1：纯数字校验
        try:
            timestamp = int(timestamp_str)
        except ValueError:
            return False

        # 条件2：长度16~19位（TUM纳秒时间戳典型范围）
        len_ts = len(timestamp_str)
        if not (16 <= len_ts <= 19):
            return False

        # 条件3：数值对应2014~2030年（避免异常值）
        min_ns = 1388534400000000000  # 2014-01-01 00:00:00 的纳秒时间戳
        max_ns = 1893456000000000000  # 2030-01-01 00:00:00 的纳秒时间戳
        if not (min_ns <= timestamp <= max_ns):
            return False

        return True
    # -----------------------------------------------------------------------------

    # 1. 验证输入文件有效性（基础校验，避免操作错误路径）
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"轨迹文件不存在: {pose_path}")
    if not os.path.isfile(pose_path):
        raise IsADirectoryError(f"输入路径不是文件: {pose_path}")
    # 额外校验：文件是否可读写（避免权限问题导致覆盖失败）
    if not os.access(pose_path, os.R_OK):
        raise PermissionError(f"无读取权限: {pose_path}")
    if not os.access(pose_path, os.W_OK):
        raise PermissionError(f"无写入权限（无法覆盖）: {pose_path}")

    # 2. 备份原文件（默认开启，覆盖前必须备份，防止转换错误导致数据丢失）
    if backup_original:
        backup_path = f"{pose_path}.backup"
        # 若备份已存在，提示用户（避免静默覆盖备份）
        if os.path.exists(backup_path):
            print(f"警告：备份文件 {backup_path} 已存在，将跳过备份（防止原备份被覆盖）")
        else:
            try:
                shutil.copy2(pose_path, backup_path)  # copy2保留文件元信息（如修改时间）
                print(f"已备份原文件至: {backup_path}")
            except Exception as e:
                raise IOError(f"备份原文件失败: {str(e)}") from e

    # 3. 核心逻辑：用临时文件存储转换后的数据，最终覆盖源文件
    # （直接读写源文件可能导致中途错误时文件损坏，用临时文件更安全）
    converted_count = 0  # 统计转换行数
    temp_file_fd, temp_file_path = tempfile.mkstemp(
        dir=os.path.dirname(pose_path),  # 临时文件放在源文件同目录，避免跨分区移动
        prefix="tum_convert_temp_",       # 临时文件前缀，便于识别
        text=True                         # 文本模式（避免二进制模式的换行符问题）
    )
    os.close(temp_file_fd)  # 关闭文件描述符，后续用with语句重新打开

    try:
        # 3.1 读取源文件 → 转换 → 写入临时文件
        with open(pose_path, 'r', encoding='utf-8') as f_in, \
                open(temp_file_path, 'w', encoding='utf-8') as f_temp:

            for line_num, line in enumerate(f_in, 1):
                stripped_line = line.strip()

                # 跳过空行和注释行（保留原格式，包括换行符）
                if not stripped_line or stripped_line.startswith('#'):
                    f_temp.write(line)
                    continue

                # 分割行数据（TUM格式：空格/制表符分隔，至少7列）
                parts = stripped_line.split()
                if len(parts) < 7:
                    raise ValueError(
                        f"文件第{line_num}行格式异常（列数不足7列）: {stripped_line}\n"
                        "TUM轨迹文件每行需包含：时间戳 + 3个平移值 + 3个四元数值"
                    )

                timestamp_str = parts[0]
                # 3.2 判断是否为纳秒，仅转换纳秒格式
                if is_nanosecond_timestamp(timestamp_str):
                    # 计算小数点位置（1秒=1e9纳秒 → 长度-9）
                    decimal_pos = len(timestamp_str) - 9
                    decimal_pos = max(1, decimal_pos)  # 避免小数点在首位（极端情况）
                    # 转换为秒格式（如1403636579763555584 → 1403636579.763555584）
                    timestamp_sec = f"{timestamp_str[:decimal_pos]}.{timestamp_str[decimal_pos:]}"
                    # 重组该行并写入临时文件
                    converted_parts = [timestamp_sec] + parts[1:]
                    f_temp.write(' '.join(converted_parts) + '\n')
                    converted_count += 1
                else:
                    # 非纳秒格式，保留原行（统一换行符为\n，避免跨平台问题）
                    f_temp.write(stripped_line + '\n')
                    #print(f"第{line_num}行：非纳秒时间戳，跳过转换 → {timestamp_str}")

        # 3.3 用临时文件覆盖源文件（shutil.move原子操作，比直接写更安全）
        shutil.move(temp_file_path, pose_path)
        #print(f"\n文件覆盖完成！源文件 {pose_path} 已更新")

    except Exception as e:
        # 若中途出错，删除临时文件，避免残留
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise RuntimeError(f"转换过程出错（源文件未被修改）: {str(e)}") from e

    # 4. 输出统计信息（让用户明确转换效果）
    # 统计有效行数（排除空行和注释行）
    with open(pose_path, 'r', encoding='utf-8') as f_in:
        total_valid_lines = sum(
            1 for line in f_in
            if line.strip() and not line.strip().startswith('#')
        )
    #print(f"统计：共{total_valid_lines}行有效轨迹数据，其中{converted_count}行完成纳秒→秒转换，{total_valid_lines - converted_count}行跳过")

    return pose_path

def evaluate_single_run(result_path, gt_path, plot=False):

    pose_path = os.path.join(result_path, "CameraTrajectory_TUM_imu.txt")

    pose_path = convert_tum_timestamp_to_seconds(pose_path)

    traj_est = file_interface.read_tum_trajectory_file(pose_path)
    gt_file = os.path.join(gt_path, 'state_groundtruth_estimate0/data.csv')
    traj_ref = file_interface.read_euroc_csv_trajectory(gt_file)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.1)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                          pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    result_rotation_part = main_ape.ape(traj_ref, traj_est, est_name='rot', pose_relation=PoseRelation.rotation_part,
                                        align=True, correct_scale=True)

    out_path=os.path.join(result_path, "metrics_traj.txt")
    with open(out_path, 'w') as fp:
        fp.write(result.pretty_str())
        fp.write(result_rotation_part.pretty_str())

    if plot:
        from evo.tools import plot
        from evo.tools.plot import PlotMode
        import matplotlib.pyplot as plt
        import copy
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_ref, correct_scale=True)
        fig = plt.figure()
        traj_by_label = {
            "estimate (not aligned)": traj_est,
            "estimate (aligned)": traj_est_aligned,
            "reference": traj_ref
        }
        plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        plt.show()

    return result.stats


def batch_eval_sequences(sequence_dir, gt_path):
    """
    批量评估序列并在命令行显示结果表格
    参数:
        sequence_dir: 所有序列的估计结果根目录
        gt_path: 数据集GT根目录
    """
    # 1. 定义待评估的所有序列（与你的Euroc序列匹配）
    sequences = [
        "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
        "V1_01_easy", "V1_02_medium", "V1_03_difficult",
        "V2_01_easy", "V2_02_medium", "V2_03_difficult"
    ]

    # 2. 收集所有序列的评估结果
    all_results = {}
    print("=" * 80)
    print("开始批量评估Euroc序列...")
    print("=" * 80)

    for seq in sequences:
        # 拼接当前序列的估计路径和GT路径（适配Euroc目录结构）
        est_seq_dir = os.path.join(sequence_dir, seq)
        gt_seq_dir = os.path.join(gt_path, seq, "mav0")  # Euroc GT默认在mav0子目录

        # 路径有效性检查（跳过不存在的序列）
        if not os.path.exists(est_seq_dir):
            print(f"⚠️  跳过 {seq}：估计结果目录不存在 → {est_seq_dir}")
            continue
        if not os.path.exists(gt_seq_dir):
            print(f"⚠️  跳过 {seq}：GT目录不存在 → {gt_seq_dir}")
            continue

        # 执行单序列评估
        metrics = evaluate_single_run(est_seq_dir, gt_seq_dir, plot=True)
        all_results[seq] = metrics
        print(f"✅ {seq} 评估完成 ")


    # 无有效结果时直接返回
    if not all_results:
        print("\n❌ 未获取到任何有效评估结果！")
        return

    # 3. 提取指标列名（取第一个有效结果的指标键）
    metric_names = list(next(iter(all_results.values())).keys())

    # 4. 在命令行打印格式化表格
    print("\n" + "=" * 100)
    # 表格表头（左对齐，固定宽度）
    print(f"{'序列名':<20}" + " | ".join([f"{metric:<12}" for metric in metric_names]))
    print("=" * 100)
    # 表格内容（逐行打印每个序列的指标值）
    for seq, metrics in all_results.items():
        metric_vals = [f"{metrics[metric]:<12.4f}" for metric in metric_names]
        print(f"{seq:<20}" + " | ".join(metric_vals))
    print("=" * 100)

    # 5. （可选）打印所有序列的指标平均值（参考用）
    print("\n📊 所有有效序列指标平均值：")
    print("-" * 50)
    for metric in metric_names:
        avg_val = sum([res[metric] for res in all_results.values()]) / len(all_results)
        print(f"{metric:<12}: {avg_val:.4f}")
    print("=" * 50)



# 使用示例
if __name__ == "__main__":
    # 配置路径
    sequence_dir = "/home/zhiyu/PHD_Code/CVPR2026/AERGS-SLAM/results/euroc_dark"  # revise to your path
    gt_path = "/media/zhiyu/Seagate/Dataset/euroc"                                # revise to your euroc dataset path
    batch_eval_sequences(sequence_dir, gt_path)
