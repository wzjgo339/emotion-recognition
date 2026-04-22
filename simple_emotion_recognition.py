#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的表情识别 - 直接使用最佳配置
一键识别，无需选择
"""

import os
import sys
import glob
from pathlib import Path

def run_command(cmd):
    """执行命令"""
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='gbk', errors='replace')
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("[表情识别] 表情识别 - 简化版")
    print("=" * 50)
    print("使用最佳模型: best_fixed_model.pth")
    print("使用增强推理: simple_enhanced_inference.py")
    print("默认不截取图片（适合手工裁剪的图像）")
    print("=" * 50)
    
    # 检查最佳模型是否存在
    best_model = "best_fixed_model.pth"
    if not os.path.exists(best_model):
        print(f"[错误] 最佳模型不存在: {best_model}")
        print("正在运行快速修复...")
        success, stdout, stderr = run_command("python quick_fix.py")
        if success:
            print("[成功] 模型修复完成！")
        else:
            print("[失败] 模型修复失败")
            return
    
    print("\n[菜单] 选择操作:")
    print("1. [单图] 识别单张图片")
    print("2. [批量] 批量识别文件夹")
    print("3. [评估] 显示模型评估结果")
    print("4. [退出] 退出程序")
    
    while True:
        try:
            choice = input("\n请选择 (1-4): ").strip()
            if choice not in ['1', '2', '3', '4']:
                print("[错误] 请输入1-4")
                continue
            
            if choice == '1':
                # 单张图片识别
                print("\n[单图] 获取图片...")
                
                # 获取当前目录的图片
                current_images = list(Path('.').glob('*.jpg')) + list(Path('.').glob('*.png'))
                
                if current_images:
                    print("发现的图片:")
                    for i, img in enumerate(current_images[:10], 1):
                        print(f"  {i}. {img.name}")
                    print(f"  0. 输入其他路径")
                    
                    try:
                        img_choice = input(f"请选择 (0-{len(current_images)}): ").strip()
                        if img_choice == '0':
                            image_path = input("请输入图片路径: ").strip().strip('"\'')
                        else:
                            img_idx = int(img_choice) - 1
                            if 0 <= img_idx < len(current_images):
                                image_path = str(current_images[img_idx])
                            else:
                                print("[错误] 无效选择")
                                continue
                    except ValueError:
                        print("[错误] 请输入有效数字")
                        continue
                else:
                    image_path = input("请输入图片路径: ").strip().strip('"\'')
                
                if not os.path.exists(image_path):
                    print("[错误] 图片不存在")
                    continue
                
                # 询问是否需要可视化
                try:
                    visualize_choice = input("是否生成可视化结果? (y/n): ").strip().lower()
                    visualize = visualize_choice in ['y', 'yes', '是', '']
                except:
                    visualize = False
                
                visualize_flag = ' --visualize' if visualize else ''
                cmd = f'python simple_enhanced_inference.py --image "{image_path}" --model {best_model}{visualize_flag}'
                run_inference(cmd)
                print("\n[完成] 单图识别完成，返回主菜单...")
                
            elif choice == '2':
                # 批量识别
                folder_path = input("请输入文件夹路径: ").strip().strip('"\'')
                if not os.path.isdir(folder_path):
                    print("[错误] 文件夹不存在")
                    continue
                
                cmd = f'python simple_enhanced_inference.py --batch "{folder_path}" --model {best_model} --visualize'
                run_inference(cmd)
                print("\n[完成] 批量识别完成，返回主菜单...")
                
            elif choice == '3':
                # 显示模型评估结果
                print("\n[评估] 显示模型评估结果...")
                print("正在生成和显示模型在测试集上的评估报告...")
                
                # 先运行评估（如果还没有结果）
                if not os.path.exists('evaluation_results'):
                    print("首次运行，正在生成评估结果...")
                    cmd = 'D:\\anaconda3\\envs\\myPytorch\\python.exe evaluate_model.py --model best_fixed_model.pth'
                    run_inference(cmd)
                
                # 显示可视化结果
                print("\n正在显示评估结果的可视化图表...")
                cmd = 'D:\\anaconda3\\envs\\myPytorch\\python.exe show_results.py'
                run_inference(cmd)
                print("\n[完成] 模型评估完成，返回主菜单...")
                
            elif choice == '4':
                print("[再见] 程序退出！")
                sys.exit(0)
                
        except KeyboardInterrupt:
            print("\n[再见] 程序退出！")
            sys.exit(0)

def run_inference(cmd):
    """执行推理"""
    print(f"\n[执行] {cmd}")
    print("=" * 60)
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("[成功] 执行成功!")
        if stdout:
            print("\n[结果]:")
            print(stdout)
    else:
        print("[失败] 执行失败!")
        if stderr:
            print("\n[错误]:")
            print(stderr)
    
    print("=" * 60)

if __name__ == '__main__':
    main()