import cv2
import numpy as np
import math


class Point:
    x = 0
    y = 0


# 旋转图像中的点
def get_rotate_point(src_image, points, rotate_center, src_angle):
    dst_points = Point()
    row = src_image.shape[0]

    x1 = points.x
    y1 = row - points.y
    x2 = rotate_center.x
    y2 = row - rotate_center.y
    x = np.round(
        (x1 - x2) * math.cos(math.pi / 180.0 * src_angle) - (y1 - y2) * math.sin(math.pi / 180.0 * src_angle) + x2)
    y = np.round(
        (x1 - x2) * math.sin(math.pi / 180.0 * src_angle) + (y1 - y2) * math.cos(math.pi / 180.0 * src_angle) + y2)
    y = row - y
    dst_points.x = int(x)
    dst_points.y = int(y)
    dst_points1 = [0] * 2
    dst_points1[0] = dst_points.x
    dst_points1[1] = dst_points.y
    dst_points1 = tuple(dst_points1)
    cv2.circle(src_image, dst_points1, 5, (255, 0, 0), 2)

    return dst_points


def star_six(src, pa, pb, dis):  # 返回值类型为point2i
    row = src.rows  # 需要将坐标系左上角转化到左下角
    ya = row - pa.y
    yb = row - pb.y

    k = (ya - yb) / (pa.x - pb.x)
    b = yb - k * pb.x
    star = Point()

    if k < 0:
        delta0 = math.pi * 1.0 + math.atan(k)
    else:
        delta0 = math.atan(k)

    # 求角度，代入三角函数计算的必须是弧度，而不是角度

    if pa.x < pb.x:
        star.x = pa.x - math.fabs(3 * dis * math.cos(delta0))  # 偏移量为星杆长在水平方向投影
        star.y = row - (k * star.x + b)

    if pa.x > pb.x:
        star.x = pa.x + math.fabs(3 * dis * math.cos(delta0))
        star.y = row - (k * star.x + b)

    if (pa.x == pb.x) and (ya > yb):
        star.x = pa.x
        star.y = row - (ya - 3 * dis)
    if (pa.x == pb.x) and (ya > yb):
        star.x = pa.x
        star.y = row - (ya + 3 * dis)
    return star


# 确定螺旋桨桨心区域的ROI
def roi(image, po, dist, k):
    pi = Point()
    length = int(1.0 * dist)
    width = int(0.8 * dist)

    pi.x = int(np.round(po.x - length / 2))
    pi.y = int(np.round(po.y - width / 2))
    # ROI越界问题
    if pi.x < 0:
        pi.x = 0
    elif pi.y < 0:
        pi.y = 0
    elif pi.x + length > image.shape[1]:
        pi.x = image.shape[1] - length
    elif pi.y + width > image.shape[0]:
        pi.y = image.shape[0] - width
    # img_roi = cv2.rectangle(image, (pi.x, pi.y), (pi.x+length, pi.y+width), (255, 0, 0), 2)
    # 修改参数
    if k == 1:
        img_roi = image[pi.y:pi.y + width, pi.x + 20:pi.x + 20 + length]
    else:
        img_roi = image[pi.y:pi.y + width, pi.x:pi.x + length]
    return img_roi


# 确定1和6螺旋桨桨心区域的ROI
def roi_16(image, po, dist, k):
    pi = Point()
    length = int(1.0 * dist)
    width = int(0.8 * dist)

    if k == 0:
        pi.x = int(np.round(po.x - 0.9 * length))
        pi.y = int(np.round(po.y - 0.5 * width))

    if k == 5:
        pi.x = int(np.round(po.x - 0.1 * length))
        pi.y = int(np.round(po.y - 0.5 * width))

    if pi.x < 0:
        pi.x = 0

    if pi.y < 0:
        pi.y = 0

    if pi.x + length > image.shape[1]:
        pi.x = image.shape[1] - length

    if pi.y + width > image.shape[0]:
        pi.y = image.shape[0] - width
    # img_roi = cv2.rectangle(image, (pi.x, pi.y), (pi.x+length, pi.y+width), (255, 0, 0), 2)
    # 修改参数
    if k == 0:
        img_roi = image[pi.y + 10:pi.y + 10 + width, pi.x + 50:pi.x + 50 + length]
    else:
        img_roi = image[pi.y:pi.y + width, pi.x:pi.x + length]
    return img_roi


# 求红绿标记的斜率倾角
def angle(src, pa, pb):
    row = src.shape[0]

    # 需要将坐标系左上角转化到左下角
    ya = row - pa.y
    yb = row - pb.y
    k = (ya - yb) / (pa.x - pb.x)
    if k < 0:
        delta0 = math.pi * 1.0 + math.atan(k)
    else:
        delta0 = math.atan(k)
    delta = (delta0 * 180.0 / math.pi)
    return delta


# 确定整体轮廓
def location_all(image_src):
    image = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(image, image, (3, 3), 5)
    # 转为2值图
    cv2.adaptiveThreshold(image, image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 3)
    cv2.medianBlur(image, image, 11)  # 中值滤波
    cv2.imshow("中值", image)
    cv2.Canny(image, image, 50, 200, 3)  # 进行canny边缘检测，为霍夫变换做准备
    cv2.imshow("边缘检测", image)
    # 查找轮廓
    contours = np.array([], [])
    cv2.findContours(image, contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 在黑布上绘制轮廓
    result = np.zeros(image.size(), dtype=np.uint8)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
    result_line = np.zeros(image.size(), dtype=np.uint8)  # 定义绘制直线的白布
    result_line.fill(255)
    src_image = result.copy()
    lines = cv2.HoughLinesP(src_image, 1, math.pi / 180, 50, 30, 5)
    for q in lines[0]:
        cv2.line(result_line, (lines[q][0], lines[q][1]), (lines[q][2], lines[q][3]), (0, 0, 0), 2, 8, 0)
        cv2.imshow("直线", result_line)


# 确定颜色标记块的圆心坐标，并进行排序
def red_center(i_low_h, i_high_h, i_low_s, i_high_s, i_low_v, i_high_v, img_original):
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
    # 开始检测不规则红色块儿
    # 边缘检测
    img_thr = cv2.Canny(img_thr, 50, 100)
    # 检测并绘制轮廓
    contours, hierarchy = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 查找出所有的圆边界
    result = np.zeros(img_thr.shape, np.uint8)
    for e in range(0, len(contours)):  # 遍历所有顶层的轮廓
        cv2.drawContours(result, contours, -1, 255)  # 只检测到一条轮廓

    drawing = np.zeros(result.shape, np.uint8)  # 定义黑色画布
    min_rect = []

    for k in contours:
        min_rect.append(cv2.minAreaRect(k))  # 输出是矩形的四个点坐标
        box = cv2.boxPoints(cv2.minAreaRect(k))
        box = np.int0(box)
        # 代入cvMinAreaRect2这个函数得到最小包围矩形  这里已得出被测物体的角度，宽度,高度，和中点坐标点存放在CvBox2D类型的结构体中，
        # 主要工作基本结束
        cv2.drawContours(drawing, contours, -1, 255)
        # rotated rectangle
        cv2.drawContours(result, [box], 0, (0, 0, 255), 3)
    point00 = []
    for counter_1 in range(0, len(min_rect)):
        point00.append(min_rect[counter_1][0])

    for list_counter in range(0, len(point00)):
        point00[list_counter] = list(point00[list_counter])
    # 纵坐标升序排列
    for w in range(0, len(point00)):
        for s in range(len(point00) - 1, w, -1):
            if point00[w][1] > point00[s][1]:
                temp1 = point00[w][1]
                z1 = point00[w][0]
                point00[w][1] = point00[s][1]
                point00[w][0] = point00[s][0]
                point00[s][1] = temp1
                point00[s][0] = z1

    return point00


def perspective(original_image):
    point0 = red_center(160, 179, 90, 255, 90, 255, original_image)
    point004 = [0] * 4
    point004[0] = point0[0]
    point004[1] = point0[1]
    point004[2] = point0[2]
    point004[3] = point0[3]
    for w in range(0, len(point004)):
        for s in range(len(point004) - 1, w, -1):
            if point004[w][0] > point004[s][0]:
                temp_2 = point004[w][0]
                z_2 = point004[w][1]
                point004[w][0] = point004[s][0]
                point004[w][1] = point004[s][1]
                point004[s][0] = temp_2
                point004[s][1] = z_2

        # 定义目标图像perspective image.

    # 取四个梯形点
    image_points = np.zeros((4, 2))
    image_points[0][0] = point004[0][0]
    image_points[0][1] = point004[0][1]
    image_points[1][1] = point004[1][1]
    image_points[1][0] = point004[1][0]
    image_points[2][0] = point004[2][0]
    image_points[2][1] = point004[2][1]
    image_points[3][0] = point004[3][0]
    image_points[3][1] = point004[3][1]
    objective_points = np.zeros((4, 2))

    la = image_points[0][0] - 10 - (image_points[0][0] + 10)
    lb = 0.866 * la
    objective_points[0][0] = (image_points[0][0] + 10)
    objective_points[0][1] = (0.5 * original_image.shape[1])
    objective_points[1][0] = (objective_points[0][0] + 0.5 * la)
    objective_points[1][1] = (objective_points[0][1] - lb)
    objective_points[2][0] = (objective_points[0][0] + 1.5 * la)
    objective_points[2][1] = (objective_points[0][1] - lb)
    objective_points[3][0] = (objective_points[0][0] + 2.0 * la)
    objective_points[3][1] = (objective_points[0][1])
    per_src = np.float32(
        [[float(objective_points[0][0]), float(objective_points[0][1])],
         [float(objective_points[1][0]), float(objective_points[1][1])],
         [float(objective_points[2][0]), float(objective_points[2][1])],
         [float(objective_points[3][0]), float(objective_points[3][1])]])
    per_dst = np.float32(
        [[float(image_points[0][0]), float(image_points[0][1])], [float(image_points[1][0]), float(image_points[1][1])],
         [float(image_points[2][0]), float(image_points[2][1])],
         [float(image_points[3][0]), float(image_points[3][1])]])
    transform = cv2.getPerspectiveTransform(per_dst, per_src)
    # 仿射变换以3个点为基准点，即使数组长度为4也仅取前3个点作为基准点
    # 透视变换以4个点为基准点，两种变换结果不相同。
    perspective_image = cv2.warpPerspective(original_image, transform,
                                            (original_image.shape[0], original_image.shape[1]),
                                            cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, cv2.BORDER_REPLICATE)
    cv2.imshow("11111", perspective_image)
    return perspective_image


# 确定支杆和桨叶线的斜率
def location(src_image, angle_biaoji, j0):
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    image1 = cv2.GaussianBlur(gray, (3, 3), 5)
    # 转为2值图
    image2 = cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 3)
    image3 = cv2.medianBlur(image2, 9)  # 中值滤波
    # Mat midImage;
    image4 = cv2.Canny(image3, 50, 200, 3)  # 进行canny边缘检测，为霍夫变换做准备
    # 字符串转化为文件名
    cv2.imshow("erzhi" + str(j0 + 1), image4)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(image4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 在黑布上绘制轮廓
    result = np.zeros(image4.shape, dtype=np.uint8)
    result.fill(0)
    cv2.drawContours(result, contours, -1, 255, 2)
    # cv2.imshow("countor" + str(j0 + 1), result)
    result_line = np.zeros(image4.shape, dtype=np.uint8)  # 定义绘制直线的白布
    result_line.fill(255)
    src_image2 = result.copy()
    # 修改参数
    lines1 = cv2.HoughLinesP(src_image2, 1, np.pi / 180, 60, minLineLength=30, maxLineGap=5)

    angle_list = []
    for t in range(0, len(lines1)):
        cv2.line(result_line, (lines1[t][0][0], lines1[t][0][1]), (lines1[t][0][2], lines1[t][0][3]), (0, 0, 0), 2)
        first = Point()
        second = Point()
        first.x = lines1[t][0][0]
        first.y = lines1[t][0][1]
        second.x = lines1[t][0][2]
        second.y = lines1[t][0][3]
        ang = angle(src_image2, first, second)
        angle_list.append(ang)
        if ang > 175 or ang < 5:
            ang = 0
            print(ang)
        angle_list.append(ang)
        if angle_list is None:
            print("vector is empty!")
    cv2.imshow("line" + str(j0 + 1), result_line)
    angle_list.sort(reverse=False)
    n = len(angle_list)
    m_min = (angle_list[0] + angle_list[1] + angle_list[2]) / 3.0
    m_max = (angle_list[n - 1] + angle_list[n - 2] + angle_list[n - 3]) / 3.0
    print("mMax: %f mMin: %f" % (m_max, m_min))
    print("angleBiaoji:" + str(angle_biaoji))
    a = abs(m_max - angle_biaoji)
    b = abs(m_min - angle_biaoji)
    # 由于1和6号桨叶的ROI与其他区域不同，因此需要单独考虑，直接用标志线获取差角
    if j0 == 0 or j0 == 5:

        if a > b:
            return m_max - angle_biaoji
        else:
            return m_min - angle_biaoji

    else:
        if a > b:
            return m_max - m_min
        else:
            return m_min - m_max


img1 = cv2.imread("C:\\pictureSource\\picture1.jpg")
img2 = perspective(img1)
img_a = cv2.resize(img2, (960, 1280))

point002 = []
point1 = red_center(160, 179, 90, 255, 90, 255, img_a)  # 颜色标记初始化
# 在透射变换之后的图片中
if point1[2][0] > point1[3][0]:
    temp2 = point1[2][0]
    z2 = point1[2][1]
    point1[2][0] = point1[3][0]
    point1[2][1] = point1[3][1]
    point1[3][0] = temp2
    point1[3][1] = z2

blue_center = [0] * 2
green_center = [0] * 2
blue_center[0] = int(np.round(point1[2][0]))
blue_center[1] = int(np.round(point1[2][1]))
green_center[0] = int(np.round(0.5 * (0.5 * (point1[6][0] + point1[7][0]) + 0.5 * (point1[4][0] + point1[5][0]))))
green_center[1] = int(np.round(point1[6][1]))
blue_center = tuple(blue_center)
green_center = tuple(green_center)

cv2.circle(img_a, green_center, 2, (255, 0, 0), 2)
cv2.circle(img_a, blue_center, 2, (0, 0, 255), 2)

# 标志连线的斜率
distance = math.sqrt(
    (blue_center[0] - green_center[0]) * (blue_center[0] - green_center[0]) + (blue_center[1] - green_center[1]) * (
            blue_center[1] - green_center[1]))
print(distance)


p1 = Point()
p2 = Point()
p1.x = blue_center[0]
p1.y = blue_center[1]
dstROI = []
angANG = []
rotate_green = Point()
rotate_green.x = green_center[0]
rotate_green.y = green_center[1]
rotate_blue = Point()
rotate_blue.x = blue_center[0]
rotate_blue.y = blue_center[1]
blue_center = list(blue_center)
angle_point = Point()
for i in [0, 1, 2, 3, 4, 5]:
    P2 = get_rotate_point(img_a, rotate_blue, rotate_green, i * 60)

    angle_point.x = P2.x
    angle_point.y = P2.y
    biaoji_angle = angle(img_a, angle_point, rotate_green)
    if i == 0 or i == 5:
        dst = roi_16(img_a, angle_point, distance, i)
    else:
        dst = roi(img_a, angle_point, distance, i)
    cv2.imshow("1" + str(i), dst)
    dstROI.append(dst)
    if i == 1:
        biaoji_angle = biaoji_angle + 10
    elif i == 4:
        biaoji_angle = biaoji_angle - 10 + 180
    angANG.append(biaoji_angle)
print("各个桨叶从第一个开始，依次旋转角度如下：")
for j in [0, 1, 2, 3, 4, 5]:
    print("第" + str(j) + "个桨叶")
    fai = np.round(location(dstROI[j], angANG[j], j))
    if fai > 0:
        if fai > 90:
            fai = fai - 90
            print("第 " + str(j + 1) + "个桨叶顺时针调整：" + str(fai) + " 度")
        else:
            fai = 90 - fai
            print("第 " + str(j + 1) + "个桨叶逆时针调整：" + str(fai) + " 度")
    else:
        if fai < -90:
            fai = -90 - fai
            print("第 " + str(j + 1) + "个桨叶逆时针调整：" + str(fai) + " 度")
        else:
            fai = fai + 90
            print("第 " + str(j + 1) + "个桨叶顺时针调整：" + str(fai) + " 度")

cv2.waitKey(0)
