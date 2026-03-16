def generate_exposure_factors(input_filename, output_filename):
    """
    读取时间戳文件，为每个时间戳生成随机曝光因子并保存

    参数:
        input_filename: 输入的时间戳文件名
        output_filename: 输出的包含曝光因子的文件名
    """
    # 确保随机数生成器的一致性（可选）
    # random.seed(42)

    try:
        # 读取时间戳文件
        with open(input_filename, 'r') as f_in:
            timestamps = f_in.readlines()

        # 处理每个时间戳，生成曝光因子
        results = []
        for timestamp in timestamps:
            # 去除换行符和空白字符
            timestamp = timestamp.strip()
            if timestamp:  # 跳过空行
                # 生成0.3到2.0之间的均匀分布随机数，保留两位小数
                exposure_factor = round(random.uniform(0.5, 1.5), 2)
                results.append(f"{timestamp} {exposure_factor}\n")

        # 保存结果到输出文件
        with open(output_filename, 'w') as f_out:
            f_out.writelines(results)

        print(f"已处理: {input_filename} -> {output_filename}，共 {len(results)} 条记录")

    except Exception as e:
        print(f"处理文件 {input_filename} 时出错: {str(e)}")

def main():
    # 定义要处理的11个Euroc序列
    sequences = [
        "MH01", "MH02", "MH03", "MH04", "MH05",
        "V101", "V102", "V103",
        "V201", "V202", "V203"
    ]

    # 获取当前目录下的所有文件
    current_dir_files = set(os.listdir('.'))

    # 处理每个序列
    for seq in sequences:
        input_file = f"{seq}.txt"
        output_file = f"{seq}_factor.txt"

        # 检查输入文件是否存在
        if input_file in current_dir_files:
            generate_exposure_factors(input_file, output_file)
        else:
            print(f"警告: 文件 {input_file} 不存在，已跳过")

if __name__ == "__main__":
    main()
