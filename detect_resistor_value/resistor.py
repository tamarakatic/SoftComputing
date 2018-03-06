import numpy as np
import cv2
import math

from sklearn.cluster import KMeans

for idx in range(1, 5):
    img = cv2.imread('images/resistor{}.png'.format(idx))

    kernel_3 = np.ones((3, 3), np.uint8)

    median = cv2.medianBlur(img, 5)
    blur = cv2.bilateralFilter(median, 5, 110, 120)

    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel_3)

    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_3)
    kernel_4 = np.ones((4, 4), np.uint8)

    erosion = cv2.erode(opening, kernel_4, iterations=1)

    edges = cv2.Canny(erosion, 15, 80, 3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)

    for rho, theta in lines[0]:
        print("rho: {} theta: {}".format(rho, theta))
        num_rows, num_cols = img.shape[:2]
        print("rows: {} cols: {} ".format(num_rows, num_cols))
        degree = math.degrees(theta)

        angle = degree - 90
        print(angle)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        alpha = 1000
        x1 = int(x0 + alpha * (-b))
        y1 = int(y0 + alpha * (a))
        x2 = int(x0 - alpha * (-b))
        y2 = int(y0 - alpha * (a))

        xn = np.cos(math.radians(angle)) * x0 - \
            np.sin(math.radians(angle)) * y0
        yn = np.sin(math.radians(angle)) * x0 + \
            np.cos(math.radians(angle)) * y0
        print("xn: {} yn: {}".format(xn, yn))

        img_cpy = img.copy()
        cv2.line(img_cpy, (x1, y1), (x2, y2), (255, 255, 255), 90)
        print("x1: {} y1: {} x2: {} y2: {}".format(x1, y1, x2, y2))

        rotation_matrix = cv2.getRotationMatrix2D(
            (num_cols / 2, num_rows / 2), angle, 1)
        rot_img = cv2.warpAffine(
            img_cpy, rotation_matrix, (num_cols, num_rows))
        org = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))

        gray = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
        mask = gray < 255
        gray[mask] = 0
        binary = gray

    _, contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def find_largest_cnt(contours):
        max_area = 0
        largest_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                largest_contour = cnt
        return largest_contour

    x, y, w, h = cv2.boundingRect(find_largest_cnt(contours))
    cropped_org = org[y:y + h, x:x + w]
    cropped = cv2.medianBlur(cropped_org, 11)
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)

    k1 = 0.5
    k2 = 6.5

    def color_std(lab_img):
        stds_col = np.zeros((100, lab_img.shape[1]), dtype=np.uint8)
        stds_row = np.zeros((lab_img.shape[0], 50), dtype=np.uint8)
        L, a, b = cv2.split(lab_img)

        for col in range(lab_img.shape[1]):
            std_L = np.std(L[:, col])
            std_a = np.std(a[:, col])
            std_b = np.std(b[:, col])
            stds_col[:, col] = int(k1 * std_L + k2 * (std_a + std_b))

        for row in range(lab_img.shape[0]):
            std_L = np.std(L[row, :])
            std_a = np.std(a[row, :])
            std_b = np.std(b[row, :])
            stds_row[row, :] = int(k1 * std_L + k2 * (std_a + std_b))

        return stds_col, stds_row

    std_col, std_row = color_std(lab)

    std_row = cv2.equalizeHist(std_row)

    std_col_bgr = cv2.cvtColor(std_col, cv2.COLOR_GRAY2BGR)
    std_row_bgr = cv2.cvtColor(std_row, cv2.COLOR_GRAY2BGR)

    right_part = np.vstack(
        (std_row_bgr, np.zeros((100, 50, 3), dtype=np.uint8)))
    left_part = np.vstack((cropped, std_col_bgr))
    out_image = np.hstack((left_part, right_part))

    _, std_col_thresh = cv2.threshold(std_col, 70, 255, cv2.THRESH_BINARY)
    kernel_6 = np.ones((6, 6), np.uint8)
    dilation_col = cv2.dilate(std_col_thresh, kernel_6, iterations=1)
    closing_col = cv2.morphologyEx(dilation_col, cv2.MORPH_CLOSE, kernel_6)

    _, cnt_std_col, _ = cv2.findContours(
        closing_col, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x1, y1, w1, h1 = cv2.boundingRect(find_largest_cnt(cnt_std_col))
    print("x1: {} 1: {}".format(x1, y1))

    _, std_row_thresh = cv2.threshold(std_row, 120, 255, cv2.THRESH_BINARY)

    _, cnt_std_row, _ = cv2.findContours(
        std_row_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x2, y2, w2, h2 = cv2.boundingRect(find_largest_cnt(cnt_std_row))
    cropped_row = cropped_org[y2:y2 + h2, x1:x1 + w1]

    row, col, _ = cropped_row.shape
    y1 = int(row * 0.1)
    y2 = int(row * 0.9)
    x1 = int(col * 0.15)
    x2 = int(col * 0.85)

    cropped_cpy = cropped_row[y1:y2, x1:x2]

    eroded = cv2.erode(cropped_cpy, (5, 5), iterations=3)

    img = np.zeros(eroded.shape, dtype=np.uint8)
    b, g, r = cv2.split(eroded)

    color_mean_b = np.mean(b, axis=0)
    color_mean_g = np.mean(g, axis=0)
    color_mean_r = np.mean(r, axis=0)

    img[:, :, 0] = color_mean_b
    img[:, :, 1] = color_mean_g
    img[:, :, 2] = color_mean_r

    h, w = img.shape[:2]

    image = img.reshape((img.shape[0] * img.shape[1], 3))

    kmeans = KMeans(n_clusters=5)
    labels = kmeans.fit_predict(image)

    quant = kmeans.cluster_centers_.astype("uint8")[labels]

    from collections import Counter
    cnt = Counter(labels)

    most_common_label, _ = cnt.most_common()[0]

    most_common_rgb = kmeans.cluster_centers_.astype("uint8")[
        most_common_label]

    quant = quant.reshape((h, w, 3))
    image = quant.copy()

    eroded = cv2.erode(quant, (5, 5), iterations=1)

    quant[np.where((quant != most_common_rgb).all(axis=2))] = [255, 255, 255]
    quant[np.where((quant == most_common_rgb).all(axis=2))] = [0, 0, 0]

    binary = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)

    _, contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects = sorted(rects, key=lambda x: x[0])

    cv2.imshow("image", np.vstack([image, quant]))
    cv2.waitKey(0)

    i = 0
    for ind, rect in enumerate(rects):
        x, y, w, h = rect
        print("x: {} y: {} w: {} h: {}".format(x, y, w, h))
        if i < 4:
            crop = image[y:h+y, x:w+x]
            i += 1

            h, w = crop.shape[:2]
            crop_color = crop[h / 2, w / 2]

            cv2.imshow("crop", crop)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
