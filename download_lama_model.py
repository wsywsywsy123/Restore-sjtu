#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaMa AI修复模型下载脚本
从Hugging Face下载 LaMa ONNX 模型文件
"""

import os
import sys
import requests
import io

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    import codecs
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def download_file(url, filename):
    """
    下载文件并显示进度条
    """
    try:
        print(f"正在从 {url} 下载模型文件...")
        print(f"保存到: {filename}")
        
        # 发送GET请求，stream=True用于流式下载
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 获取文件总大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 创建进度条
        if HAS_TQDM:
            with open(filename, 'wb') as f, tqdm(
                desc=os.path.basename(filename),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        else:
            # 没有tqdm时使用简单进度显示
            downloaded = 0
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
            print()  # 换行
        
        print(f"\n[成功] 下载完成！文件已保存到: {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n[错误] 下载失败: {e}")
        return False
    except Exception as e:
        print(f"\n[错误] 发生错误: {e}")
        return False

def main():
    """
    主函数
    """
    # 模型文件信息
    model_filename = "lama-fcn_resolution_robust_512.onnx"
    
    # GitHub 仓库可能的模型文件URL（尝试多个可能的文件名）
    # 从 advimman/lama 仓库的 models 目录下载
    github_urls = [
        "https://github.com/advimman/lama/raw/main/models/lama-fcn_resolution_robust_512.onnx",
        "https://github.com/advimman/lama/raw/main/models/big_lama.onnx",
        "https://github.com/advimman/lama/raw/main/models/lama.onnx",
    ]
    
    # 备用：Hugging Face 下载链接
    huggingface_url = "https://huggingface.co/shzym/lama-fcn/resolve/main/lama-fcn_resolution_robust_512.onnx"
    
    # 尝试的下载URL列表（按优先级排序）
    download_urls = github_urls + [huggingface_url]
    
    print("=" * 60)
    print("LaMa AI修复模型下载工具")
    print("=" * 60)
    print()
    print("正在尝试从多个源下载模型...")
    print("主要来源: https://github.com/advimman/lama/tree/main/models")
    print()
    print("注意：")
    print("- GitHub 仓库可能包含 PyTorch 模型文件（.pth/.ckpt），需要转换为 ONNX")
    print("- 如果找不到 ONNX 文件，您可以：")
    print("  1. 手动从仓库下载 PyTorch 模型并转换")
    print("  2. 使用现有的多尺度修复方法（无需模型）")
    print()
    
    # 检查文件是否已存在
    if os.path.exists(model_filename):
        print(f"[警告] 文件 {model_filename} 已存在！")
        choice = input("是否重新下载？(y/n): ").strip().lower()
        if choice != 'y':
            print("取消下载。")
            return
        # 删除旧文件
        try:
            os.remove(model_filename)
            print(f"已删除旧文件: {model_filename}")
        except Exception as e:
            print(f"删除旧文件失败: {e}")
            return
    
    # 检查是否安装了必要的库
    try:
        import requests
    except ImportError:
        print("[错误] 需要安装 requests 库")
        print("请运行: pip install requests")
        return
    
    # 开始下载（尝试多个URL）
    print()
    success = False
    for i, url in enumerate(download_urls, 1):
        filename_from_url = url.split('/')[-1]
        print(f"尝试下载源 {i}/{len(download_urls)}: {filename_from_url}")
        # 如果URL中的文件名不同，先尝试用原文件名，失败后再用目标文件名
        temp_success = download_file(url, filename_from_url)
        if temp_success:
            # 如果下载成功但文件名不同，重命名
            if filename_from_url != model_filename:
                try:
                    if os.path.exists(model_filename):
                        os.remove(model_filename)
                    os.rename(filename_from_url, model_filename)
                    print(f"已将文件重命名为: {model_filename}")
                except Exception as e:
                    print(f"重命名失败: {e}，但文件已下载为: {filename_from_url}")
                    print(f"请手动重命名为: {model_filename}")
            success = True
            break
        else:
            if i < len(download_urls):
                print(f"下载源 {i} 失败，尝试下一个...")
                print()
    
    if success:
        # 验证文件大小（模型文件大约50MB）
        file_size = os.path.getsize(model_filename)
        print(f"\n文件大小: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1024 * 1024:  # 小于1MB可能下载不完整
            print("[警告] 文件大小异常小，可能下载不完整！")
            print("请检查网络连接或手动下载。")
        else:
            print("\n[成功] 模型文件下载成功！")
            print(f"现在可以在 app.py 中使用 LaMa AI 修复功能了。")
    else:
        print("\n[错误] 所有下载源都失败了！")
        print("\n您可以尝试以下方法：")
        print("1. 检查网络连接")
        print("2. 手动访问 GitHub 仓库下载：")
        print("   https://github.com/advimman/lama/tree/main/models")
        print("   找到 ONNX 模型文件，点击文件名进入详情页")
        print("   点击 'Download' 或 'Raw' 按钮下载")
        print("3. 或者访问 Hugging Face：")
        print("   https://huggingface.co/shzym/lama-fcn/tree/main")
        print("4. 将下载的文件放在与 app.py 相同的目录下")
        print("5. 确保文件名是: " + model_filename)
        print("\n重要提示：")
        print("- GitHub 仓库 (https://github.com/advimman/lama) 主要提供 PyTorch 模型")
        print("- 您需要将 PyTorch 模型转换为 ONNX 格式才能使用")
        print("- 或者，您可以继续使用现有的多尺度修复方法（无需模型）")
        print("\n转换 PyTorch 到 ONNX 的方法：")
        print("1. 安装: pip install torch onnx")
        print("2. 使用 LaMa 官方提供的转换脚本")
        print("3. 或者使用其他在线转换工具")
        print("\n建议：")
        print("如果没有 ONNX 模型，应用会自动使用多尺度修复方法，效果也很好！")

if __name__ == "__main__":
    main()

