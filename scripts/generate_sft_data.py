from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from random import seed
from random import randint
from typing import List

import torch

device = "npu:0"

model_path = "/root/projects/ascend_r1_turtorial/output/qwen-3b-r1-coundown"
### 模型本地部署
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device
)


###  生成提示词模板
def generate_r1_prompt(numbers, target):
    """
    生成 R1 Countdown 游戏提示词

    参数:
        numbers (list[int]): 数字列表
        target (int): 目标值
    返回:
        dict: 生成的一个数据样本
    """
    # 定义提示词前缀
    r1_prefix = [
        {
            "role": "user",
            "content": f"使用给定的数字 {numbers}，创建一个等于 {target} 的方程。你可以使用基本算术运算（+、-、*、/）一次或多次，但每个数字只能使用一次。在 <think> </think> 标签中展示你的思考过程，并在 <answer> </answer> 标签中返回最终方程，例如 <answer> (1 + 2) / 3 </answer>。在 <think> 标签中逐步思考。",
        },
        {
            "role": "assistant",
            "content": "让我们逐步解决这个问题。\n<think>",  # 结尾使用 `<think>` 促使模型开始思考
        },
    ]

    return tokenizer.apply_chat_template(
        r1_prefix, tokenize=False, continue_final_message=True)


### 奖励函数筛选格式和答案正确的结果
def format_reward_func(completion):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs

      Returns:
          float: Reward score
    """
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

        match = re.search(regex, completion, re.DOTALL)
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards = 0.0
        else:
            rewards = 1.0
    except Exception:
        rewards = 0.0
    return rewards


def equation_reward_func(completion, gt, numbers):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (str): Generated outputs
        target (str): Expected answers
        nums (list[str]): Available numbers

    Returns:
        float: Reward scores
    """
    try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards = 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            rewards = 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
            rewards = 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builti'ns__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:
            rewards = 1.0
        else:
            rewards = 0.0
    except Exception:
        # If evaluation fails, reward is 0
        rewards = 0.0
    return rewards


import re

### 批量运行脚本
import jsonlines
from tqdm import tqdm


## 随机生成数据脚本
def gen_dataset(
        num_samples: int,
        num_operands: int = 3,
        min_target:int=10,
        max_target: int = 20,
        min_number: int = 1,
        max_number: int = 10,
        operations: List[str] = ['+', '-', '*', '/'],
        seed_value: int = 20024,
) -> List[dict]:
    """Generate dataset for countdown task.

    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        operations: List of allowed operations
        seed_value: Random seed for reproducibility

    Returns:
        List of tuples containing (target, numbers, solution)
    """
    seed(seed_value)
    samples = []

    for _ in tqdm(range(num_samples)):
        # Generate random target
        target = randint(min_target, max_target)

        # Generate random numbers
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]

        results = {
            "target": target,
            "nums": numbers
        }

        samples.append(results)

    return samples


def data_batch_generation(output_file="/root/test/sft_data.jsonl"):
    # 生成随机数据
    ds = gen_dataset(num_samples=50)

    # 批量运行数据
    with jsonlines.open(output_file, 'a') as writer:
        for idx, data in tqdm(enumerate(ds), total=len(ds), desc="Processing data"):

            target = data['target']
            nums = data['nums']

            # 合成prompt
            prompt = generate_r1_prompt(nums, target)

            # 推理部分
            model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 最终输出
            question = """使用给定的数字 {nums}，创建一个等于 {target} 的方程。你可以使用基本算术运算（+、-、*、/）一次或多次，但每个数字只能使用一次。在 <think> </think> 标签中展示你的思考过程，并在 <answer> </answer> 标签中返回最终方程，例如 <answer> (1 + 2) / 3 </answer>。在 <think> 标签中逐步思考。让我们逐步解决这个问题。\n<think>"""
            question = question.format(nums=nums, target=target)

            # 每10条数据打印一次results
            if (idx % 1 == 0):  # 假设数据集中有一个 'id' 字段
                print(response)

            # 奖励分数判断
            format_reward = format_reward_func(response)
            if format_reward == 0.0:
                continue

            equation_reward = equation_reward_func(response, target, nums)
            if equation_reward == 0.0:
                continue

            results = {
                "input": question,
                "output": response
            }

            # 每10条数据打印一次results
            if (idx % 1000 == 0):  # 假设数据集中有一个 'id' 字段
                print(results)

            # 保存成jsonl文件
            writer.write(results)



if __name__=="__main__":
    output_path = "./sftp_data.jsonl"
    data_batch_generation(output_file=output_path)