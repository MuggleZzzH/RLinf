import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到 PYTHONPATH
REPO_PATH = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_PATH))

from rlinf.data.datasets.recap.value_model import ValueDataset

def test_dataset():
    # 使用你其中一个 SFT 数据集进行测试
    dataset_path = "/mnt/public/songsiqi/data/lerobot/beijing_guqiuyi_20260317_pm_tele_s2m"
    tag = "turtle2_v1"  # 刚才 Step 1 生成的 tag
    
    print(f"正在测试数据集: {dataset_path}")
    print(f"使用 Tag: {tag}")
    
    try:
        # 实例化 ValueDataset
        # 注意：这里会触发我们刚刚修改的 robot_type='turtle2' 逻辑
        ds = ValueDataset(
            dataset_path=dataset_path,
            robot_type="turtle2",
            model_type="pi05",
            tag=tag,
            action_horizon=10,
            action_dim=32,
            normalize_to_minus_one_zero=True,
            max_samples=5 # 只加载几个样本
        )
        
        print(f"数据集加载成功，样本总数: {len(ds)}")
        
        # 获取第一个样本
        sample = ds[0]
        
        print("\n--- 样本检查 ---")
        print(f"Keys in sample: {list(sample.keys())}")
        
        # 1. 检查图像
        images = sample.get("images", {})
        print(f"\n图像检查 (Camera Views): {list(images.keys())}")
        expected_cameras = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for cam in expected_cameras:
            if cam in images:
                img = images[cam]
                # X2RobotInputs 应该输出 uint8 (H, W, C)
                # 注意：ValueDataset 在最终返回前可能已经处理过，或者保留原始格式
                print(f"  - {cam}: shape={getattr(img, 'shape', 'N/A')}, type={type(img)}")
            else:
                print(f"  - [错误] 缺失摄像头: {cam}")

        # 2. 检查 Prompt
        print(f"\nPrompt: {sample.get('prompt')}")
        
        # 3. 检查 Return
        print(f"Target Value (Return): {sample.get('target_values')}")
        
        print("\n[成功] 链路检查完成！你的 X1 机器人适配逻辑已成功跑通。")
        
    except Exception as e:
        print(f"\n[失败] 发现错误:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
