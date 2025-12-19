"""
@filename:herbal_show.py
@author:Jason
@time:2025-12-14
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def create_comparison_data(width=600, height=400):
    """
    【模拟数据生成】
    构造一个具有挑战性的场景：两个紧挨着的孔洞。
    """
    # 1. 背景
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # 模拟切片 (浅黄色)
    cv2.circle(img, (300, 200), 160, (220, 245, 255), -1)

    # --- 关键场景：两个紧挨着的孔洞 ---
    # 孔洞 A
    # h1_center = (250, 200)
    # cv2.circle(img, h1_center, 30, (20, 20, 20), -1)

    # 孔洞 B (圆心距离很近，边缘会相交/粘连)
    h2_center = (300, 200)
    cv2.circle(img, h2_center, 30, (20, 20, 20), -1)

    # --- 3. 生成 Mask ---

    # A. 语义分割 Mask (Semantic)
    # 逻辑：只管类别。孔洞A是类别1，孔洞B也是类别1。
    # 结果：它们像素值相同，融合在一起。
    sem_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(sem_mask, h1_center, 30, 1, -1)  # 类别 1
    cv2.circle(sem_mask, h2_center, 30, 1, -1)  # 类别 1

    # B. 实例分割 Mask (Instance)
    # 逻辑：管个体。孔洞A是实例ID 1，孔洞B是实例ID 2。
    # 结果：它们像素值不同，界限分明。
    inst_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(inst_mask, h1_center, 30, 1, -1)  # 实例 ID 1
    cv2.circle(inst_mask, h2_center, 30, 2, -1)  # 实例 ID 2 (覆盖了重叠区域的一部分，体现独立性)

    return img, sem_mask, inst_mask


def colorize_mask(mask, mode='semantic'):
    """
    上色函数
    mode='semantic': 使用固定类别颜色 (所有孔洞都是红色)
    mode='instance': 使用随机个体颜色 (孔洞A紫色，孔洞B橙色)
    """
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(mask)

    # 语义分割标准色卡 (BGR格式)
    # 1=孔洞(红)
    semantic_palette = {
        0: (0, 0, 0),
        1: (255, 0, 0)  # 红色
    }

    for uid in unique_ids:
        if uid == 0: continue

        if mode == 'semantic':
            color = semantic_palette.get(uid, (255, 255, 255))
        else:
            # 实例分割：根据ID生成伪随机颜色
            # 这种技巧可以保证同一个ID每次生成的颜色是一样的，但不同ID颜色不同
            np.random.seed(int(uid) * 100)
            color = np.random.randint(50, 255, size=3).tolist()

        mask_region = (mask == uid)
        color_img[mask_region] = color

    return color_img


def overlay(image, mask_rgb, alpha=0.7):
    return cv2.addWeighted(image, 1, mask_rgb, alpha, 0)


def main():
    # 1. 生成数据
    raw_img, sem_mask, inst_mask = create_comparison_data()

    # 转RGB用于显示
    raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # 2. 处理语义分割图
    sem_color = colorize_mask(sem_mask, mode='semantic')
    sem_overlay = overlay(raw_img_rgb, sem_color)

    # 3. 处理实例分割图
    inst_color = colorize_mask(inst_mask, mode='instance')
    inst_overlay = overlay(raw_img_rgb, inst_color)

    # --- 绘图对比 ---
    plt.figure(figsize=(15, 5))

    # 图1：原图
    plt.subplot(1, 3, 1)
    plt.title("Original Image\n(Two touching holes)", fontsize=12)
    plt.imshow(raw_img_rgb)
    plt.axis('off')

    # 图2：语义分割
    plt.subplot(1, 3, 2)
    plt.title("Semantic Segmentation\n(Class Aware only)", fontsize=12)
    plt.imshow(sem_overlay)
    plt.xlabel("Result: 1 big merged blob")  # 标注
    plt.axis('off')

    # 图3：实例分割
    plt.subplot(1, 3, 3)
    plt.title("Instance Segmentation\n(Object Aware)", fontsize=12)
    plt.imshow(inst_overlay)
    plt.xlabel("Result: 2 distinct objects")  # 标注
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()