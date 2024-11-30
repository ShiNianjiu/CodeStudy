import os
import urllib.request

import json
import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm


# 下载并加载指定大小的GPT-2模型
def download_and_load_gpt2(model_size, models_dir):
    allowed_sizes = ("124M", "355M", "774M", "1558M")  # GPT-2模型支持的尺寸
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")  # 如果指定尺寸不在允许范围内，抛出异常

    model_dir = os.path.join(models_dir, model_size)  # 模型的存储目录
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 模型文件的下载地址
    filenames = [  # 模型所需的所有文件
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    os.makedirs(model_dir, exist_ok=True)  # 创建模型目录，如果已存在则不报错
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)  # 文件的完整URL
        file_path = os.path.join(model_dir, filename)  # 文件在本地存储的路径
        download_file(file_url, file_path)  # 下载文件

    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取TensorFlow检查点的路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载模型设置
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 从TensorFlow检查点加载GPT-2参数

    return settings, params  # 返回模型设置和参数


# 下载文件，带有进度条显示
def download_file(url, destination):
    try:
        with urllib.request.urlopen(url) as response:  # 使用urllib打开URL
            file_size = int(response.headers.get("Content-Length", 0))  # 获取文件大小

            if os.path.exists(destination):  # 如果文件已存在
                file_size_local = os.path.getsize(destination)  # 获取本地文件大小
                if file_size == file_size_local:  # 如果文件大小相同
                    print(f"File already exists and is up-to-date: {destination}")  # 打印提示信息
                    return  # 返回，不继续下载

            block_size = 1024  # 设置块大小

            progress_bar_description = os.path.basename(url)  # 获取URL的文件名作为进度条描述
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:  # 初始化进度条
                with open(destination, "wb") as file:  # 以二进制写模式打开目标文件
                    while True:
                        chunk = response.read(block_size)  # 读取一块数据
                        if not chunk:
                            break  # 如果数据已读完，则退出循环
                        file.write(chunk)  # 将数据写入文件
                        progress_bar.update(len(chunk))  # 更新进度条
    except urllib.error.HTTPError:  # 如果发生HTTP错误
        s = (
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)  # 打印错误信息

# 注意：原代码中`download_file`函数被重复定义了，这里仅保留了一个版本，并添加了中文注释。
# 如果在实际使用中，需要确保不要重复定义同一个函数。

# 从TensorFlow检查点加载GPT-2参数
def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}  # 初始化参数字典，包含所有层的空字典

    for name, _ in tf.train.list_variables(ckpt_path):  # 遍历检查点中的所有变量
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))  # 加载变量并去除多余的维度

        variable_name_parts = name.split("/")[1:]  # 将变量名按"/"分割成部分

        target_dict = params  # 目标字典初始化为params
        if variable_name_parts[0].startswith("h"):  # 如果变量名以"h"开头，表示是隐藏层的变量
            layer_number = int(variable_name_parts[0][1:])  # 获取层号
            target_dict = params["blocks"][layer_number]  # 将目标字典设置为对应层的字典

        for key in variable_name_parts[1:-1]:  # 遍历变量名的中间部分，构建目标字典的路径
            target_dict = target_dict.setdefault(key, {})  # 如果键不存在，则创建空字典并赋值

        last_key = variable_name_parts[-1]  # 获取变量名的最后一部分作为键
        target_dict[last_key] = variable_array  # 将变量数组赋值给目标字典的对应键

    return params  # 返回构建好的参数字典