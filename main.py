import glob
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import time

templates_files = glob.glob("templates/*.png")
templates = [cv.imread(f, cv.IMREAD_GRAYSCALE) for f in templates_files]

device = 0
width = 640
height = 480
cap = cv.VideoCapture(device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)


# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

def get_templates():
    template_list = list()
    templates_files = glob.glob("templates/*.png")
    for file in templates_files:
        template_image = cv.imread(file)
        template_list.append(template_image)
    return template_list

template_list = get_templates()

# for template in templates:
    #     for meth in methods:
    #         img = frame.copy()
    #         method = eval(meth)
    #         # Apply template Matching
    #         res = cv.matchTemplate(img, template, method)
    #         min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    #         # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    #         if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    #             top_left = min_loc
    #         else:
    #             top_left = max_loc
    #         w, h = template.shape[::-1]
    #         bottom_right = (top_left[0] + w, top_left[1] + h)
    #         cv.rectangle(img, top_left, bottom_right, 255, 2)
    #         plt.subplot(121), plt.imshow(res, cmap='gray')
    #         plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    #         plt.subplot(122), plt.imshow(img, cmap='gray')
    #         plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    #         plt.suptitle(meth)
    #         plt.show()
def my_frame_differencing(prev, curr):
    '''
    Function that does frame differencing between the current frame and the previous frame
    Args:
        src The current color image
        prev The previous color image
    Returns:
        dst The destination grayscale image where pixels are colored white if the corresponding pixel intensities in the current
    and previous image are not the same
    '''
    dst = cv.absdiff(prev, curr)
    gs = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    dst = (gs > 50).astype(np.uint8) * 255
    return dst



def mask(frame):
    # 创建了一个基于K近邻算法的背景减法器对象,背景减法器用于从视频中提取前景对象，即将移动的对象与静止的背景进行分离
    knn = cv.createBackgroundSubtractorKNN()

    frame_mask = knn.apply(frame)
    # 使用中值滤波器对前景掩码进行中值模糊处理，去除噪点
    frame_mask = cv.medianBlur(frame_mask, 3)
    # 使用均值滤波器对前景掩码进行模糊处理，进一步平滑化图像。
    frame_mask = cv.blur(frame_mask, (3, 3))
    # 对前景掩码进行膨胀操作，用于连接物体的边界，填充空洞。
    frame_mask = cv.dilate(frame_mask, np.ones((7, 7)))
    # 将前景掩码中大于0的像素值设为255，其余像素值设为0，将其转换为8位无符号整数类型。这样做的目的可能是将前景掩码二值化。
    frame_mask = ((frame_mask > 0) * 255).astype(np.uint8)
    # cv.imshow("frame_mask", frame_mask)
    return frame_mask

def skin(frame, frame_mask):
    image = np.array(frame)
    # 将图像从BGR颜色空间转换为HSV（色相、饱和度、明度）颜色空间。
    imageHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # 数对两个阈值化后的图像进行按位或操作
    skinMaskHSV = cv.bitwise_or((cv.inRange(imageHSV, (0, 25, 80), (50, 255, 255))),
                                 (cv.inRange(imageHSV, (150, 25, 80), (255, 255, 255))))

    skinHSV = cv.bitwise_and(image, image, mask=skinMaskHSV)

    # 对皮肤检测后的图像进行中值滤波处理,以减少图像中的噪声和细节，并且能够保留图像的边缘信息
    skinMatching = cv.medianBlur(skinHSV, 5)

    # 将图像从BGR颜色空间转换为灰度（单通道）图像
    skinMatching = cv.cvtColor(skinMatching, cv.COLOR_BGR2GRAY)

    # 将灰度图像进行二值化处理，将大于0的像素值设为255，其余像素值设为0，并将结果转换为8位无符号整数类型（uint8）
    skinMatching = ((skinMatching > 0) * 255).astype(np.uint8)
    # cv.imshow("skin match", skinMaskHSV)
    return skinMatching


def matchTemplate(img, templates, titles, method=cv.TM_CCORR_NORMED):

    found = None
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        r = img.shape[1] / float(resized.shape[1])
        # edged = cv2.Canny(resized, 40, 90)
        edged = resized
        # cv2.imshow('frame', edged)
        for template, t in zip(templates, titles):
            tH, tW = template.shape[:2]
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            # cv2.imshow("Template", template)

            result = cv.matchTemplate(edged, template, method)
            (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r, t)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (maxVal, maxLoc, r, t) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    return maxVal, t, startX, startY, endX, endY

colors = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (230, 245, 66)
}
ts = glob.glob("templates3/*.png")

ts_gray = [cv.imread(t, cv.IMREAD_GRAYSCALE) for t in ts]
ns = [(os.path.basename(t)).split(".")[0] for t in ts]
fourcc = cv.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv.CAP_PROP_FPS)
output = cv.VideoWriter('output.mp4',fourcc, fps, (width * 3 // 2, height))
cv.namedWindow("hand reg", cv.WINDOW_AUTOSIZE)

while cap.isOpened():
    start = int(round(time.time() * 1000))
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = mask(frame)
    skinmatch = skin(frame, fgmask)
    fgmask = cv.bitwise_and(skinmatch, fgmask)
    # cv.imshow("result", fgmask)
    tt = np.zeros_like(fgmask)
    contours, hierarchy = cv.findContours(fgmask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        maxCont = max(contours, key=cv.contourArea)
        blankImage = np.zeros_like(fgmask)
        cv.drawContours(blankImage, [maxCont], -1, (255, 255, 255), thickness=cv.FILLED)
        maxVal, name, startX, startY, endX, endY = matchTemplate(blankImage, ts_gray, ns)
        print(maxVal, name, startX, startY, endX, endY)
        if maxVal > 0.7:
            box = [(startX, startY), (endX, endY)]
            cv.putText(frame, name, (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv.rectangle(frame, box[0], box[1], (255, 255, 255), 2)
            idx = ns.index(name)
            tH, tW = ts_gray[idx].shape[:2]
            tt = cv.copyMakeBorder(ts_gray[idx], 50, height - tH - 50, 50, width - tW - 50, cv.BORDER_CONSTANT,
                                    value=0)
        v_comb = np.vstack([blankImage, tt])
    else:
        v_comb = np.vstack([fgmask, tt])

    v_comb = cv.resize(v_comb, (0, 0), fx=0.5, fy=0.5)
    v_comb = cv.cvtColor(v_comb, cv.COLOR_GRAY2BGR)
    combined = np.hstack([frame, v_comb])
    cv.imshow("hand reg", combined)
    # video_shower.frame = combined
    output.write(combined)
    end = int(round(time.time() * 1000)) - start
    print("fps=", (1000.0 / end))
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.imwrite('skinDetection.png', blankImage)
        break

cap.release()
output.release()
cv.destroyAllWindows()


if __name__ == '__main__':
    print("quit")
