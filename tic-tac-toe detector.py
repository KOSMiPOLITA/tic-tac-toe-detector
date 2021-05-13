import cv2
import imutils
import statistics
import numpy as np
import math

#do image podajemy ścieżkę, w której znajduje się obraz

image = cv2.imread("Kolko_i_krzyzyk-main/tic_tac_toe_dedector/105.jpg")

kernel = np.ones((2, 2), np.uint8)
iq = image.copy()



def get_central_square_lines(img):
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1]), reverse=True)

    for (i,c) in enumerate(cnts[0:1]):
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        hull = cv2.convexHull(approx)
    hull1 = np.ndarray((8,1,2), dtype=np.int32)
    if len(hull) > 8:
        ind = 0
        ind1 = 0
        for i in hull:
            add = True
            for j in hull[ind+1:]:

                if (abs(i[0][0] - j[0][0]) < 20) and (abs(i[0][1] - j[0][1]) < 20) and ((i[0][0] != j[0][0]) \
                        or (i[0][1] != j[0][1])) :
                    add = False
            if add == True:
                hull1[ind1][0][0] = np.int32(i[0][0])
                hull1[ind1][0][1] = np.int32(i[0][1])
                ind1+=1
            ind +=1
        return hull1

    return hull


def get_angle(hull):

    y_max = hull[hull[:,0,0].argsort()[::-1][:2]]
    y_min = hull[hull[:, 0, 0].argsort()[:2]]
    x_max = hull[hull[:, 0, 1].argsort()[::-1][:2]]
    x_min = hull[hull[:, 0, 1].argsort()[:2]]
    y_max = y_max[y_max[:, 0, 1].argsort()]
    y_min = y_min[y_min[:, 0, 1].argsort()]
    x_max =x_max[x_max[:, 0, 0].argsort()]
    x_min =x_min[x_min[:, 0, 0].argsort()]

    a = np.arctan((y_max[1][0][0] - y_min[1][0][0])/(y_max[1][0][1] - y_min[1][0][1]))
    a = np.degrees(abs(a))
    tmp = np.arctan((y_max[0][0][0] - y_min[0][0][0]) / (y_max[0][0][1] - y_min[0][0][1]))
    tmp = np.degrees(abs(tmp))
    a = (a+tmp)/2

    return a

def points(hull):
    y_max = hull[hull[:,0,0].argsort()[::-1][:2]]
    y_min = hull[hull[:, 0, 0].argsort()[:2]]
    x_max = hull[hull[:, 0, 1].argsort()[::-1][:2]]

    x_min = hull[hull[:, 0, 1].argsort()[:2]]
    y_max = y_max[y_max[:, 0, 1].argsort()]
    y_min = y_min[y_min[:, 0, 1].argsort()]
    x_max =x_max[x_max[:, 0, 0].argsort()]
    x_min =x_min[x_min[:, 0, 0].argsort()]

    a1 = ((y_max[0][0][0] - y_min[0][0][0])) / ((y_max[0][0][1] - y_min[0][0][1]))
    b1 = (y_min[0][0][1]- (a1 * y_min[0][0][0]))
    p1 = line_intersection((y_max[0][0], y_min[0][0]), (x_min[0][0], x_max[0][0]))
    p2 =line_intersection((y_max[0][0], y_min[0][0]), (x_min[1][0], x_max[1][0]))
    p3 = line_intersection((y_max[1][0], y_min[1][0]), (x_min[0][0], x_max[0][0]))
    p4 = line_intersection((y_max[1][0], y_min[1][0]), (x_min[1][0], x_max[1][0]))

    bo = np.array([p1, p2, p3, p4], dtype=np.int)
    for i in bo:
        if i[0] == -1 and i[1] == -1:
            return []
        i[0] = int(i[0])
        i[1] = int(i[1])

    return bo

def w_h(hull):
    y_max = hull[hull[:,0,0].argsort()[::-1][:2]]
    y_min = hull[hull[:, 0, 0].argsort()[:2]]
    x_max = hull[hull[:, 0, 1].argsort()[::-1][:2]]

    x_min = hull[hull[:, 0, 1].argsort()[:2]]
    y_max = y_max[y_max[:, 0, 1].argsort()]
    y_min = y_min[y_min[:, 0, 1].argsort()]
    x_max =x_max[x_max[:, 0, 0].argsort()]
    x_min =x_min[x_min[:, 0, 0].argsort()]

    a1 = ((y_max[0][0][0] - y_min[0][0][0])) / ((y_max[0][0][1] - y_min[0][0][1]))
    b1 = (y_min[0][0][1]- (a1 * y_min[0][0][0]))

    p1 = line_intersection((y_max[0][0], y_min[0][0]), (x_min[0][0], x_max[0][0]))
    p2 =line_intersection((y_max[0][0], y_min[0][0]), (x_min[1][0], x_max[1][0]))
    p3 = line_intersection((y_max[1][0], y_min[1][0]), (x_min[0][0], x_max[0][0]))
    p4 = line_intersection((y_max[1][0], y_min[1][0]), (x_min[1][0], x_max[1][0]))
    bo = np.array([p1, p2, p3, p4])
    h = (len_between_points(p1, p2) + len_between_points(p4, p3))/2
    w =(len_between_points(p1, p3) + len_between_points(p2, p4))/2

    box = [[0],[[0],[w,h]]]

    return box

def len_between_points(p1, p2):

    x = (p1[0] - p2[0])*(p1[0] - p2[0])
    y = (p1[1] - p2[1])*(p1[1] - p2[1])
    a = math.sqrt(x+y)
    return a

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return [-1, -1]

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]




def crop_rect(img, rect, reverse = False):

    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = img.shape[0], img.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    if angle < -45:
        angle = -1 * (-90 - angle)

    if reverse == True:
        angle = -1*angle

    return img_crop, imutils.rotate_bound(img, -angle)


def get_central_square(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1]), reverse=True)

    curr_area = 0
    b = -1

    if len(cnts) > 1:
        M = cv2.moments(cnts[1])
    else:
        M = cv2.moments(cnts[0])

    cX = 0
    cY = 0
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    for (i, c) in enumerate(cnts[2:]):
        epsilon = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        approx_area = cv2.contourArea(approx)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = np.int0(box)

        if (rect[1][0] * rect[1][1]) != 0 and (approx_area / (rect[1][0] * rect[1][1])) > 0:
            if cv2.pointPolygonTest(box, (cX, cY), False) == 1 and curr_area < approx_area:
                cv2.drawContours(mask, [c], 0, (255), 1)
                cv2.drawContours(mask, [box], 0, (255), 1)
                curr_area = approx_area
                b = [box, rect, approx]

    return b


def recognize(img):
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
    char = '-'
    pX = 10000
    pY = 10000
    to_draw = []

    for (i, c) in enumerate(cnts[0:3]):
        mask1 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        cv2.drawContours(mask1, [c], -1, (255), 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask1 = cv2.dilate(mask1, kernel)
        cnts1 = cv2.findContours(mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cnts1 = imutils.grab_contours(cnts1)
        cnts1 = sorted(cnts1, key=lambda c: cv2.contourArea(c), reverse=True)
        epsilon = 0.04 * cv2.arcLength(cnts1[0], True)
        approx = cv2.approxPolyDP(cnts1[0], epsilon, True)

        M = cv2.moments(cnts[0])
        cX = 0
        cY = 0
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        approx_area = cv2.contourArea(approx)

        if cv2.pointPolygonTest(cnts1[0], (cX, cY), False) != 1:
            continue

        if abs(img.shape[0] / 2 - cX) > pX and abs(img.shape[1] / 2 - cY) > pY:
            break

        if cv2.pointPolygonTest(cnts1[0], (cX, cY), False) != 1:
            if i == 0:
                continue
            else:
                break

        hull = cv2.convexHull(approx)
        hullArea = cv2.contourArea(hull)
        r = cv2.boundingRect(c)

        if (r[2] / r[3]) > 2 or (r[3] / r[2]) > 2:
            hullArea = 0

        if (r[2] * r[3] < 0.01 * (img.shape[0] * img.shape[1])):
            hullArea = 0

        if hullArea == 0:
            solidity = 0
        else:
            solidity = approx_area / float(hullArea)

        if solidity > 0.9:
            char = "O"
            to_draw = cnts1[0]
        elif solidity > 0.2:
            char = "X"
            to_draw = cnts1[0]

        print("{} (Contour #{}) -- solidity={:.2f}".format(char, i + 1, solidity))
        cv2.drawContours(mask, cnts1[0], -1, (255), 3)
        pX = abs(img.shape[0] / 2 - cX)
        pY = abs(img.shape[1] / 2 - cY)
    cv2.putText(mask, char, (int(mask.shape[1] / 2), int(mask.shape[0] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 4)
    return [mask, char, to_draw]


def extract(img, img_col):
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cnts = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=True)

    for (i, c) in enumerate(cnts[1:]):

        cv2.drawContours(mask, [c], 0, (255), 1)

    box = get_central_square(img)
    if box == -1:
        box = get_central_square_lines(img)
        a = get_angle(box)
        img_crop, img_rot = crop_rect(img, [(0,0), (0, 0), a])
        img_crop_col, img_rot_col = crop_rect(img_col, [(0,0), (0, 0), a])
        box = get_central_square_lines(img_rot)

        angle = [(0,0), (0, 0), a]
        bo = points(box)
        if len(bo) == 0:
            return [0]
        ret = []
        b = box
        box = w_h(box)

    else:
        boxes = []
        boxes.append(((box[1][0][0], box[1][0][1]), (box[1][1][0], box[1][1][1]), box[1][2]))
        img_crop, img_rot = crop_rect(img, boxes[0])
        img_crop_col, img_rot_col = crop_rect(img_col, boxes[0])
        angle = boxes[0]
        box = get_central_square(img_rot)

        bo = cv2.boxPoints(box[1])
        bo = np.int0(bo)

    ret = []

    ret.append(img_rot[int(1.02 * (min(bo[:, 1]))): int(0.98 * (max(bo[:, 1]))),
               int(1.02 * (min(bo[:, 0]))):int(0.98 * (max(bo[:, 0])))])
    ret.append(img_rot[0:min(bo[:, 1]), min(bo[:, 0]):max(bo[:, 0])])
    ret.append(img_rot[max(bo[:, 1]):img_rot.shape[0],
               min(bo[:, 0]):max(bo[:, 0])])
    ret.append(img_rot[min(bo[:, 1]):max(bo[:, 1]),
               0:min(bo[:, 0])])
    ret.append(img_rot[0:min(bo[:, 1]),
               0:min(bo[:, 0])])
    ret.append(img_rot[max(bo[:, 1]):img_rot.shape[0],
               0:min(bo[:, 0])])
    ret.append(img_rot[min(bo[:, 1]):max(bo[:, 1]),
               max(bo[:, 0]):img_rot.shape[1]])
    ret.append(img_rot[0:min(bo[:, 1]),
               max(bo[:, 0]):img_rot.shape[1]])
    ret.append(img_rot[max(bo[:, 1]):img_rot.shape[0],
               max(bo[:, 0]):img_rot.shape[1]])
    r = []

    for i in range(0, len(ret)):
        ret[i] = recognize(ret[i])
        r.append(ret[i][1])
    gray = cv2.cvtColor(img_rot_col, cv2.COLOR_BGR2GRAY)
    gray = 255 - gray
    gray = cv2.Canny(gray, 100, 200)
    cnts = cv2.findContours(gray.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=True)
    for i, c in enumerate(cnts[0:2]):
        x, y, w, h = cv2.boundingRect(c)
        if (w/h < 2) and w!=0 and h != 0 :
            if (h/w < 2):
                cv2.drawContours(img_rot_col, [c], 0, (0, 0, 255), 3)

    color = []
    for (i, c) in enumerate(ret):
        if c[1] == 'X':
            color.append((0,255,255))
        elif c[1] == 'O':
            color.append((0, 255, 0))
        elif c[1] == '-':
            color.append((255, 0, 0))

    b = get_central_square(img_rot)
    if b == -1:
        b = get_central_square_lines(img_rot)

    cv2.drawContours(img_rot_col, [b[2]],  0, (0, 0, 255), 2)
    cv2.putText(img_rot_col, ret[0][1], (int((min(bo[:, 0])+max(bo[:,0]))/2),
                                         int((min(bo[:, 1])+max(bo[:,1]))/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                color[0], 4)

    cv2.putText(img_rot_col, ret[1][1], (int((min(bo[:, 0]) + max(bo[:, 0])) / 2),
                                         int((min(bo[:, 1]) + max(bo[:, 1])) / 2)-int(1.1 * box[1][1][0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                color[1], 4)
    cv2.putText(img_rot_col, ret[2][1], (int((min(bo[:, 0]) + max(bo[:, 0])) / 2),
                                         int((min(bo[:, 1]) + max(bo[:, 1])) / 2) + int(1.1 * box[1][1][0])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                color[2], 4)

    cv2.putText(img_rot_col, ret[3][1], (int((min(bo[:, 0]) + max(bo[:, 0])) / 2) - int(1.1 * box[1][1][1]),
                                         int((min(bo[:, 1]) + max(bo[:, 1])) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                color[3], 4)

    cv2.putText(img_rot_col, ret[4][1], (int((min(bo[:, 0]) + max(bo[:, 0])) / 2) - int(1.1 * box[1][1][1]),
                                         int((min(bo[:, 1]) + max(bo[:, 1])) / 2)- int(1.1 * box[1][1][0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                color[4], 4)


    cv2.putText(img_rot_col, ret[5][1], (int((min(bo[:, 0]) + max(bo[:, 0])) / 2) - int(1.1 * box[1][1][1]),
                                         int((min(bo[:, 1]) + max(bo[:, 1])) / 2)  + int(1.1 * box[1][1][0])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                color[5], 4)

    cv2.putText(img_rot_col, ret[6][1], (int((min(bo[:, 0]) + max(bo[:, 0])) / 2) + int(1.1 * box[1][1][1]),
                                         int((min(bo[:, 1]) + max(bo[:, 1])) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                color[6], 4)

    cv2.putText(img_rot_col, ret[7][1], (int((min(bo[:, 0]) + max(bo[:, 0])) / 2) + int(1.1 * box[1][1][1]),
                                         int((min(bo[:, 1]) + max(bo[:, 1])) / 2)- int(1.1 * box[1][1][0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                color[7], 4)

    cv2.putText(img_rot_col, ret[8][1], (int((min(bo[:, 0]) + max(bo[:, 0])) / 2) + int(1.1 * box[1][1][1]),
                                         int((min(bo[:, 1]) + max(bo[:, 1])) / 2)  + int(1.1 * box[1][1][0])),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                color[8], 4)

    img_crop_col, img_rot_col = crop_rect(img_rot_col, angle, True)
    cnts = cv2.findContours(cv2.cvtColor(img_rot_col, cv2.COLOR_BGR2GRAY).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=True)
    box = cv2.boxPoints(cv2.minAreaRect(cnts[0]))
    bo = np.int0(box)

    return [img_rot_col[int(1 * (min(bo[:, 1])))+10: int(1 * (max(bo[:, 1])))-10,
               int(1 * (min(bo[:, 0])))+10:int(1 * (max(bo[:, 0])))-10]]



def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


boards = []
boards_coords = []
boards_color = []
image1 = image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = 255 - gray
gray2 = gray.copy()
gray = cv2.Canny(gray, 100, 200)

cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=True)
print(len(cnts))
if len(cnts) < 3:
    gray= cv2.equalizeHist(gray2)
    gray = cv2.Canny(gray, 100, 200)

    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=True)

while cnts:
    box = cv2.boxPoints(cv2.minAreaRect(cnts[0]))
    box = cv2.minAreaRect(cnts[0])
    minarrect = cv2.minAreaRect(cnts[0])
    x, y, w, h = cv2.boundingRect(cnts[0])
    if (box[1][1]!= 0 and box[1][0]!=0)and (((w / h > 4 and len(cnts) > 1) or (h/w > 4 and len(cnts) > 1)) or  (
            (box[1][0] / box[1][1]) > 4 and len(cnts) > 1) or  ((box[1][1] / box[1][0]) > 4 and len(cnts) > 1)):
        x, y, w, h = cv2.boundingRect(cnts[1])
    elif w / h > 4 or h / w > 4:
        break

    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:, :] = 255
    mask[y:y + h, x:x + w] = 0
    t = 1

    if w * h > 5000:
        boards.append(gray[int(0.6 * y):y + int(h * 1.4), int(0.6 * x):int(x + 1.3 * w)])
        boards_color.append(image[int(0.6 * y):y + int(h * 1.4), int(0.6 * x):int(x + 1.3 * w)])

        boards_coords.append([y, x, h, w])
        mask[y:y + h, x:x + w] = 0

    gray = cv2.bitwise_and(gray, mask)
    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=lambda c: (cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]), reverse=True)

t = 0

gray = cv2.cvtColor(iq, cv2.COLOR_BGR2GRAY)
gray = 255 - gray
gray = cv2.Canny(gray, 100, 200)
cnts = cv2.findContours(gray.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cv2.drawContours(iq, cnts, -1, (255), 3)

for i, j, k in zip(boards, boards_coords, boards_color):
    d = 0
    for a in extract(i, k):
        cv2.imshow(str(t) + "_" + str(d), a)
        d += 1
    t += 1

cv2.waitKey(0)
