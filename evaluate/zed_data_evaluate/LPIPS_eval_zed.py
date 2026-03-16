import lpips, os, cv2, torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace


def get_png_filenames(folder_path):
    """
    读取指定文件夹中所有以.png结尾的图片文件名，去除.png后缀后排序并返回列表

    参数:
        folder_path: 文件夹路径

    返回:
        list: 处理后的文件名列表（已排序）
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise ValueError(f"文件夹不存在: {folder_path}")

    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} 不是一个文件夹")

    # 获取文件夹中所有以.png结尾的文件
    png_filenames = []
    for filename in os.listdir(folder_path):
        # 检查是否以.png结尾（不区分大小写）
        if filename.lower().endswith('.png'):
            # 去除.png后缀
            name_without_ext = filename[:-4]
            png_filenames.append(name_without_ext)

    # 排序
    png_filenames.sort()

    return png_filenames


class LPIPS:
    loss_fn_alex = None

    @staticmethod
    def calculate(img_a, img_b, rank):
        img_a, img_b = [img.permute([2, 1, 0]).unsqueeze(0) for img in [img_a, img_b]]
        if LPIPS.loss_fn_alex == None:  # lazy init
            LPIPS.loss_fn_alex = lpips.LPIPS(net='alex', version='0.1').to(rank)
        return LPIPS.loss_fn_alex(img_a.to(rank), img_b.to(rank)).item()


def main(result_path):
    render_path = os.path.join(result_path, 'cache/render')
    gt_path = os.path.join(result_path, 'cache/gt')

    # 验证路径存在性
    for path in [render_path, gt_path]:
        if not os.path.exists(path):
            raise ValueError(f"路径不存在: {path}")

    # 获取所有PNG文件名
    files = get_png_filenames(render_path)
    if not files:
        print("没有找到PNG文件")
        return

    lpips_list = []
    lpips_file_path = os.path.join(result_path, "lpips.txt")

    # 使用进度条迭代处理文件
    for file in tqdm(files, desc="计算LPIPS", unit="文件"):
        # 构建完整路径（使用os.path.join确保跨平台兼容性）
        gt_img_path = os.path.join(gt_path, f"{file}.png")
        render_img_path = os.path.join(render_path, f"{file}.png")

        # 读取图像
        gt = cv2.imread(gt_img_path)
        render = cv2.imread(render_img_path)

        # 检查图像是否成功读取
        if gt is None:
            tqdm.write(f"警告: 无法读取GT图像 {gt_img_path}")  # 进度条中显示警告
            continue
        if render is None:
            tqdm.write(f"警告: 无法读取渲染图像 {render_img_path}")
            continue

        # 转换为PyTorch张量并移动到GPU
        gt_image_torch = torch.from_numpy(np.array(gt)).to("cuda") / 255.0
        render_image_torch = torch.from_numpy(np.array(render)).to("cuda") / 255.0

        # 计算LPIPS
        val_lpips = LPIPS.calculate(
            render_image_torch.type(torch.float32),
            gt_image_torch.type(torch.float32),
            render_image_torch.device
        )

        lpips_list.append(val_lpips)

        # 写入结果到文件（每次写入确保数据被保存）
        with open(lpips_file_path, 'a', encoding='utf-8') as f:  # 使用追加模式
            line = f"{file} {val_lpips:.6f}\n"
            f.write(line)
            f.flush()  # 强制写入磁盘
            os.fsync(f.fileno())  # 确保操作系统将数据写入磁盘

    # 计算并打印平均LPIPS值
    if lpips_list:
        avg_lpips = sum(lpips_list) / len(lpips_list)
        print(f"\n平均LPIPS值: {avg_lpips:.6f}")

        # 将平均值写入文件
        with open(lpips_file_path, 'a', encoding='utf-8') as f:
            f.write(f"平均LPIPS值: {avg_lpips:.6f}\n")
            f.flush()
            os.fsync(f.fileno())
    else:
        print("\n没有计算任何LPIPS值")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="LPIPS评估脚本")
    parser.add_argument("result_path", type=str, help="结果文件夹路径")
    args = parser.parse_args()
    main(args.result_path)


