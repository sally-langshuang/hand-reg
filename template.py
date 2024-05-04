import glob
import cv2
import os

templates = glob.glob("color/*.jpg")
for template in templates:
    file_name = os.path.basename(template)

    template_image = cv2.imread(template)
    gray_frame = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    # err, hand = cv2.cvtColor(gray_frame, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # print(err)
    hand = template_image[:, :, 2]
    blur = cv2.blur(hand, (5, 5))
    _, hand = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
    file_name = os.path.basename(template)
    out = os.path.join("templates2", file_name)
    # hand = cv2.bitwise_not(hand)
    cv2.imwrite(out, hand)
if __name__ == '__main__':
    print("ok")