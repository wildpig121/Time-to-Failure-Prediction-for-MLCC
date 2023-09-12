from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 创建一个白色背景的图像
image_size = (100, 100)
image = Image.new("RGB", image_size, "white")
draw = ImageDraw.Draw(image)

# 绘制笑脸图案
radius = 40
center = (image_size[0] // 2, image_size[1] // 2)
draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline="black")
# 绘制眼睛
eye_radius = 8
left_eye_center = (center[0] - 20, center[1] - 20)
right_eye_center = (center[0] + 20, center[1] - 20)
draw.ellipse((left_eye_center[0] - eye_radius, left_eye_center[1] - eye_radius,
              left_eye_center[0] + eye_radius, left_eye_center[1] + eye_radius), fill="black")
draw.ellipse((right_eye_center[0] - eye_radius, right_eye_center[1] - eye_radius,  right_eye_center[0] + eye_radius, right_eye_center[1] + eye_radius), fill="black")

# 绘制嘴巴
draw.arc((center[0] - 20, center[1] + 10, center[0] + 20, center[1] + 30), start=0, end=180, fill="black")

# 将图像转化为 numpy 数组
image_array = np.array(image)

# 提取笑脸图案的坐标
coordinates = np.argwhere(image_array == [0, 0, 0])


# 将坐标转化为数据集
smiley_face_data = coordinates
smiley_face_data = smiley_face_data[:, :2]#第三列是颜色，去掉
# 去除重复的数据，去掉颜色后会每个数据重复三次，去掉
unique_smiley_face_data = np.unique(smiley_face_data, axis=0)
# 创建一个 DataFrame
df = pd.DataFrame(unique_smiley_face_data, columns=['X', 'Y'])


# 保存 DataFrame 到 CSV 文件
df.to_csv('smiley_face_dataset.csv', index=False)

# 可视化笑脸数据集中的笑脸图案
plt.figure(figsize=(4, 4))
plt.scatter(smiley_face_data[:, 1], -smiley_face_data[:, 0], s=1)
plt.title('Smiley Face Pattern')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()


