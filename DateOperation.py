"""
@filename:DateOperation.py
@author:Jason
@time:2025-10-09
"""
# import torch
# print(torch.cuda.is_available())
import cv2
import numpy as np
import os
from tqdm import tqdm  # 进度条库

# ================= 配置区域 =================
# 1. 原始图片文件夹路径
IMG_DIR = 'images_1'

# 2. DeepLabV3+ Mask 文件夹路径
MASK_DIR = 'masks_1'
# 3. 结果保存路径
OUTPUT_DIR = 'results_1'

# 4. Mask 的文件后缀
MASK_EXT = '.jpg'


# ===========================================

def enhance_object_with_black_bg(img, mask):
    """
    功能：
    1. 提取物体
    2. 对物体进行对比度增强 (CLAHE)
    3. 将背景置为纯黑 (0,0,0)
    """
    # 1. 确保 Mask 尺寸与原图一致
    # if img.shape[:2] != mask.shape[:2]:
    #     mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 2. 二值化 Mask (确保只有 0 和 255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. 初步提取前景
    # 利用 mask 将原图背景部分变黑
    foreground = cv2.bitwise_and(img, img, mask=binary_mask)

    # 4. 前景增强 (CLAHE)
    # 转到 LAB 空间处理亮度
    lab = cv2.cvtColor(foreground, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 应用 CLAHE (限制对比度自适应直方图均衡化)
    # clipLimit 可以调整：数值越大，对比度越强，纹理越明显
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 合并并转回 BGR
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 5. 【关键步骤】再次应用 Mask
    # 颜色空间转换可能会在黑色背景（0,0,0）上引入微小的噪点数值
    # 所以这里再次强制将背景部分置为 0
    final_result = cv2.bitwise_and(enhanced_img, enhanced_img, mask=binary_mask)

    return final_result


def main():
    # 检查输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    # 获取所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_extensions)]

    print(f"共发现 {len(image_files)} 张图片，开始处理...")

    # 循环处理
    for img_name in tqdm(image_files, desc="Processing"):

        img_path = os.path.join(IMG_DIR, img_name)

        # --- 寻找对应的 Mask ---
        # 根据您的文件名规则修改这里
        file_stem = os.path.splitext(img_name)[0]
        mask_name = file_stem + MASK_EXT
        mask_path = os.path.join(MASK_DIR, mask_name)

        if not os.path.exists(mask_path):
            # 容错：尝试加 _mask 后缀的情况
            # mask_path = os.path.join(MASK_DIR, file_stem + "_mask" + MASK_EXT)
            if not os.path.exists(mask_path):
                # print(f"跳过: 缺少 Mask -> {img_name}")
                continue

        # 读取
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)  # 灰度读取

        if img is None or mask is None:
            continue

        try:
            # === 核心处理 ===
            result_img = enhance_object_with_black_bg(img, mask)

            # 保存
            save_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(save_path, result_img)

        except Exception as e:
            print(f"处理出错 {img_name}: {e}")

    print("\n所有处理完成！结果保存在:", OUTPUT_DIR)


if __name__ == "__main__":
    main()