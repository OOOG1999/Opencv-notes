## Draw contours
```
def draw_contours(image_src):
    image = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(image, image, (3, 3), 5)
    # 转为2值图
    cv2.adaptiveThreshold(image, image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 3)
    cv2.medianBlur(image, image, 11)  # 中值滤波
    cv2.imshow("median", image)
    cv2.Canny(image, image, 50, 200, 3)  # 进行canny边缘检测，为霍夫变换做准备
    cv2.imshow("canny", image)
    # 查找轮廓
    contours = np.array([], [])
    cv2.findContours(image, contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # draw the contours in black background
    result = np.zeros(image.size(), dtype=np.uint8)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
    result_line = np.zeros(image.size(), dtype=np.uint8)  # 定义绘制直线的白布
    result_line.fill(255)
    src_image = result.copy()
    lines = cv2.HoughLinesP(src_image, 1, math.pi / 180, 50, 30, 5)
    for q in lines[0]:
        cv2.line(result_line, (lines[q][0], lines[q][1]), (lines[q][2], lines[q][3]), (0, 0, 0), 2, 8, 0)
        cv2.imshow("line", result_line)
```
## Affine translation
```
//读取原图
img = cv2.imread('xxx.jpg')
rows, cols, ch = img.shape
 
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])// this is the reference matrix
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])// this is the aim
 
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
 
cv2.imshow('image', dst)
```
## Color extraction
```
img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)  # Convert imgOriginal from BGR to HSV
    # 因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
    hsv_split = cv2.split(img_hsv)
    hsv_split[2] = cv2.equalizeHist(hsv_split[2])
    img_hsv = cv2.merge(hsv_split)
    img_thr = cv2.inRange(img_hsv, (i_low_h, i_low_s, i_low_v), (i_high_h, i_high_s, i_high_v))  # Threshold

    # 开操作 (去除一些噪点)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 形态学滤波核大小，根据实际情况试出
    img_thr = cv2.morphologyEx(img_thr, cv2.MORPH_OPEN, element)
    # 闭操作 (连接一些连通域)
    img_thr = cv2.morphologyEx(img_thr, cv2.MORPH_CLOSE, element)
    #后续可描轮廓或进行位置判定
```
