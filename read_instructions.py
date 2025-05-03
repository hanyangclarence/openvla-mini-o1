import tensorflow_datasets as tfds
import tensorflow as tf
from collections import Counter

def print_language_instructions(data_path: str):
    """
    读取RLDS数据集并打印所有的language instructions
    
    Args:
        data_path: 数据集路径
    """
    # 读取数据集
    builder = tfds.builder_from_directory(data_path)
    
    # 创建feature_spec来处理不同尺寸的图像
    feature_spec = builder.info.features.copy()
    feature_spec['steps']['observation']['image'] = tf.io.FixedLenFeature([], tf.string)
    
    ds = builder.as_dataset(split='train')
    
    # 遍历数据集
    for episode in ds:
        # 获取第一个step的instruction
        first_step = next(episode['steps'].as_numpy_iterator())
        instruction = first_step['language_instruction'].decode('utf-8')
        print(f"Instruction: {instruction}")

def count_language_instructions(data_path: str):
    """
    读取RLDS数据集并统计所有language instructions的出现次数
    
    Args:
        data_path: 数据集路径
    """
    # 读取数据集
    builder = tfds.builder_from_directory(data_path)    
    ds = builder.as_dataset(split='train')
    
    # 用于存储指令的计数器
    instruction_counter = Counter()
    
    for episode in ds:
        try:
            first_step = next(episode['steps'].as_numpy_iterator())
            instruction = first_step['language_instruction'].decode('utf-8')
            instruction_counter[instruction] += 1
            print(f"Instruction: {instruction}")
        except Exception as e:
            print(f"Skipped one episode due to error: {e}")
    
    # 打印统计结果
    print("\nInstruction counts:")
    print("-" * 50)
    for instruction, count in instruction_counter.most_common():
        print(f"Count: {count:<5} | Instruction: {instruction}")
    
    # 打印总计
    total_episodes = sum(instruction_counter.values())
    unique_instructions = len(instruction_counter)
    print("\nSummary:")
    print(f"Total episodes: {total_episodes}")
    print(f"Unique instructions: {unique_instructions}")

if __name__ == "__main__":
    data_path = "/scratch/2025_03/ghr/SyntheticVLA-mini/data/1.0.0"
    count_language_instructions(data_path)