import os
import urllib.request

import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # 验证模型大小
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # 定义路径
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)  # 创建模型目录，如果已存在则不报错
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)  # 下载文件

    # 加载设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的TensorFlow检查点路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载超参数设置
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 从TensorFlow检查点加载GPT-2参数

    return settings, params

def download_file(url, destination):
    try:
        with urllib.request.urlopen(url) as response:
            # 从响应头中获取文件大小，如果不存在则默认为0
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查文件是否已存在且大小相同
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return

            # 定义读取文件的块大小
            block_size = 1024  # 1 Kilobyte

            # 初始化进度条
            progress_bar_description = os.path.basename(url)  # 从URL中提取文件名
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # 以二进制写入模式打开目标文件
                with open(destination, "wb") as file:
                    # 分块读取文件并写入目标文件
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # 更新进度条
    except urllib.error.HTTPError:
        # 处理HTTP错误，如URL不正确、无法建立网络连接或请求的文件暂时不可用
        s = (
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)

# Alternative way using `requests`

def download_file(url, destination):
    # 使用streaming模式发送GET请求以下载文件
    response = requests.get(url, stream=True)

    # 从响应头中获取文件总大小，如果不存在则默认为0
    file_size = int(response.headers.get("content-length", 0))

    # 检查文件是否已存在且大小相同
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"文件已存在且为最新版本: {destination}")
            return

    # 定义读取文件的块大小
    block_size = 1024  # 1 Kilobyte

    # 使用文件总大小初始化进度条
    # 从URL中提取文件名作为进度条描述
    progress_bar_description = url.split("/")[-1]
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 以二进制写入模式打开目标文件
        with open(destination, "wb") as file:
            # 迭代文件数据的块
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 将块写入文件


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 初始化参数字典，为每个层创建一个空的块字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除单例维度（如果有的话）
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取相关部分
        # 跳过前缀'model/'，只关注模型内部的变量结构
        variable_name_parts = name.split("/")[1:]

        # 确定变量的目标字典
        target_dict = params
        # 如果变量名以"h"开头，表示它是某个隐藏层的参数
        if variable_name_parts[0].startswith("h"):
            # 提取层号（去掉"h"前缀并转换为整数）
            layer_number = int(variable_name_parts[0][1:])
            # 将目标字典设置为对应层的字典
            target_dict = params["blocks"][layer_number]

        # 递归地访问或创建嵌套字典
        # 遍历变量名的中间部分，直到最后一个键之前的部分
        for key in variable_name_parts[1:-1]:
            # 使用setdefault方法，如果键不存在则创建它并返回空字典
            target_dict = target_dict.setdefault(key, {})

        # 将变量数组赋值给最后一个键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    # 返回包含GPT-2模型参数的字典
    return params