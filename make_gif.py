from PIL import Image
import os
# import cv2
# 定义图片文件夹路径
image_folder = 'case1_res'

# 获取所有符合命名格式的图片文件
images = [img for img in os.listdir(image_folder) if img.startswith('case1_') and img.endswith('.jpg')]
images.sort()  # 按文件名排序

# 打开所有图片并存储在一个列表中
frames = []
for i in range(len(images)):

    image = f"case1_{i}.jpg"
    img_path = os.path.join(image_folder, image)
    print(img_path)
    img = Image.open(img_path)
    frames.append(img)

    # frame = cv2.imread(img_path)
    # height, width, layers = frame.shape

# 保存为GIF
output_gif = 'output.gif'
frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, duration=33, loop=0)

print(f"GIF saved as {output_gif}")