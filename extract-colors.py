import cv2
import numpy as np
import extcolors
import os
#Open a simple image
img=cv2.imread("images/image2.png")

#converting from gbr to hsv color space
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#skin color range for hsv color space
HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255))

HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

#converting from gbr to YCbCr color space
img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#skin color range for hsv color space
YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))
YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

#merge skin detection (YCbCr and hsv)
global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
global_mask=cv2.medianBlur(global_mask,3)
global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


# cv2.imwrite("3_global_result.jpg",global_result)
masked = cv2.bitwise_or(img,img,mask = global_mask)
cv2.imwrite("masked_image.png", masked)
colors, pixel_count = extcolors.extract_from_path("masked_image.png")

# return color with highest coverage not black
for (color, number_pixel_color) in colors:
    if (color != (0,0,0)):
        print("Skin color:", color)
        break
cv2.waitKey(0)
cv2.destroyAllWindows()

os.remove("masked_image.png") 