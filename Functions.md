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
    result_line = np.zeros(image.size(), dtype=np.uint8)  
    # white line
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
## Perspective translation
We need 8 points, 4 original picture and 4 transform points

```
pts = np.float32([ [0,0],[0,1080],[1920,1080],[1920,0] ])
 
pts1 = np.float32([[100,0],[200,1080],[1720,1080],[1920,0]])
 
M = cv2.getPerspectiveTransform(pts,pts1)
 
dst = cv2.warpPerspective(a,M,(1920,1080))
```
## Color space
img_hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)  # Convert imgOriginal from BGR to HSV<br>
The second parameter: <br>
LAB：L light, A green-red, B blue-yellow<br>
HSI: H 色相 360°, S 饱和度 [0, 1] 颜色的深浅程度, I(Intensity)亮度  0% (黑色) 到 100% (白色)<br>
HSV: V 明度 V = 0黑色, S = 0白色<br>
## 直方图均衡化
```
   hsv_split = cv2.split(img_hsv)
    hsv_split[2] = cv2.equalizeHist(hsv_split[2])
    img_hsv = cv2.merge(hsv_split)
```
增强局部的对比度而不影响整体的对比度<br>
在颜色提取中增加准确率<br>
## 腐蚀和膨胀
```
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 形态学滤波核大小，根据实际情况试出
    img_thr = cv2.morphologyEx(img_thr, cv2.MORPH_OPEN, element)
    # 闭操作 (连接一些连通域)
    img_thr = cv2.morphologyEx(img_thr, cv2.MORPH_CLOSE, element)
 ```
 腐蚀：减小高亮区域
 膨胀：扩大高亮区域
 ```
 enum MorphTypes{
    MORPH_ERODE    = 0, //腐蚀
    MORPH_DILATE   = 1, //膨胀
    MORPH_OPEN     = 2, //开操作
    MORPH_CLOSE    = 3, //闭操作
    MORPH_GRADIENT = 4, //梯度操作
    MORPH_TOPHAT   = 5, //顶帽操作
    MORPH_BLACKHAT = 6, //黑帽操作
    MORPH_HITMISS  = 7  
};
 ```
## 直线检测
```
HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=None, maxLineGap=None) 

image： 必须是二值图像，推荐使用canny边缘检测的结果图像； 
rho: 线段以像素为单位的距离精度，double类型的，推荐用1.0 
theta： 线段以弧度为单位的角度精度，推荐用numpy.pi/180 
threshod: 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。根据情况推荐先用100试试
lines：这个参数的意义未知，发现不同的lines对结果没影响，但是不要忽略了它的存在 
minLineLength：线段以像素为单位的最小长度，根据应用场景设置 
maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
```
## 对比二值化
```
cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None)

src：需要进行二值化的一张灰度图像

maxValue：满足条件的像素点需要设置的灰度值。（将要设置的灰度值）

adaptiveMethod：自适应阈值算法。可选ADAPTIVE_THRESH_MEAN_C 或 ADAPTIVE_THRESH_GAUSSIAN_C

thresholdType：opencv提供的二值化方法，只能THRESH_BINARY或者THRESH_BINARY_INV

blockSize：要分成的区域大小，上面的N值，一般取奇数

C：常数，每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值，可以为负数

dst：输出图像，可以忽略
```

