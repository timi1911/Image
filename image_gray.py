import  cv2
import os


p_r = ".6\\"
p_w = "C:\\Users\\26298\\Desktop\\result\\"
image_list = [x for x in os.listdir(p_r)]

for num, img in enumerate(image_list):
    print(num, img)
    image = cv2.imread(p_r + img)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(image1, 150, 255, cv2.THRESH_BINARY)

    image_dir = p_w + str(num).zfill(4) + '.png'
    cv2.imwrite(image_dir, dst)


# image = cv2.imread("C:\\Users\\26298\\Desktop\\224x224\\1\\")
# image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # 阈值处理
# retval, dst = cv2.threshold(image1, 80, 255, cv2.THRESH_BINARY)
#
# # 图像显示
# cv2.imshow("image", image)
# cv2.imshow("dst", dst)
#
# # 等待窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()


