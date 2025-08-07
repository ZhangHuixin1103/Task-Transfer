# viescore_evaluation.py
import json
import logging
import os
import sys

from paper_implementation.imagen_museum.utils import (get_file_path,
                                                      write_entry_to_json_file)
from paper_implementation.mllm_tools.gemini import Gemini
from viescore import VIEScore

# 确保 Python 能够找到 VIEScore 的核心库
viescore_path = '/mnt/data/huixin/Task-Transfer/VIEScore'
if viescore_path not in sys.path:
    sys.path.append(viescore_path)

# 准备日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# --- 1. 初始化 MLLM 模型 ---
# 请将 'YOUR_API_KEY.env' 替换为你的 OpenAI API Key 文件路径
try:
    mllm_model = Gemini()
except FileNotFoundError:
    print("Error: API Key file not found.")
    sys.exit(1)

# --- 2. 准备输入和输出数据 ---
# 图像路径
image_A_input_path = "data/demo/deraining/2.jpg"
image_A_output_path = "data/demo/deraining/2-derain.jpg"
image_B_input_path = "data/demo/removal/1.png"
generated_output_path = "data/demo/removal/output-after.png"

# 文本提示
text_prompt_content = """
The first two images show an example of visual task transfer.
The first image is the input (a rainy scene), and the second image is the output (the de-rained scene).
The third image is a new input (a foggy scene).
Please evaluate the fourth image, which is the model's generated output for the foggy scene.
The goal is to apply a similar visual transformation from the first example to the new input.
Rate the fourth image based on two criteria:
1.  **Semantic Consistency (SC):** How well does the fourth image successfully remove the fog, similar to how the rain was removed in the example? (1-10)
2.  **Perceptual Quality (PQ):** Is the fourth image of high visual quality? (1-10)
"""

# 将所有图像路径和文本内容组合成一个单一的 Prompt
image_prompt_list = [
    image_A_input_path,
    image_A_output_path,
    image_B_input_path,
    generated_output_path
]

# --- 3. 构建完整的 Prompt 并进行评估 ---
# VIEScore 内部会读取这些文件并构建一个完整的提示
prompt = mllm_model.prepare_prompt(image_prompt_list, text_prompt_content)

# --- 4. 运行评估并处理结果 ---
print(f"Starting evaluation...")
is_verified = False
tries = 0
max_tries = 3
# The results will be saved to this file
target_file_path = "evaluation_results.json"

while not is_verified and tries < max_tries:
    try:
        result = mllm_model.get_parsed_output(prompt)
        print("Raw result from MLLM:\n", result)

        is_verified = write_entry_to_json_file(
            input_string=text_prompt_content,
            uid="custom_msdig_eval",
            prompt_input=result,
            vision_input=image_prompt_list,
            output_file_name=target_file_path,
            give_up_parsing=False
        )

    except Exception as e:
        print(f"Error during MLLM call or parsing (try {tries+1}): {e}")

    tries += 1
    if is_verified:
        print(f"Evaluation saved to {target_file_path}")
    else:
        print("Failed to get a verified result after max tries.")
