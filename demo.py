import cv2
import numpy as np
import glob
import os

# 加载输入图像和手势模板图像
input_image = cv2.imread("input_image2.jpg")
template_image = cv2.imread("gesture_template.jpg")

template_list = []
templates_files = glob.glob("templates/*.png")
for file in templates_files:
    template_image = cv2.imread(file)
    hand = template_image[:,:,2]
    blur = cv2.blur(hand,(5,5))
    _, hand = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
    file_name =os.path.basename(file)
    cv2.imshow(file_name, hand)
    cv2.imwrite(f'images/templates/{file_name}', hand)

# 将彩色图像转换为灰度图像
gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
gray_template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

# 使用matchTemplate函数在输入图像中搜索手势模板的位置
result = cv2.matchTemplate(gray_input_image, gray_template_image, cv2.TM_CCOEFF_NORMED)

# 在匹配结果中找到最大值及其位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 确定最佳匹配的位置
top_left = max_loc
bottom_right = (top_left[0] + template_image.shape[1], top_left[1] + template_image.shape[0])

# 在输入图像中绘制矩形框
cv2.rectangle(input_image, top_left, bottom_right, (0, 255, 0), 2)

# 显示结果
cv2.imshow("Detected Gesture", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

if __name__ == '__main__':
    print("ok")