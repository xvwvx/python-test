
import cv2
import numpy as np
import random
import os
import time
import operator
import copy

def show(img):
    cv2.imshow("image", img)
    cv2.waitKey(30)
    # cv2.destroyAllWindows()

# 生成透视变换矩阵
def get_matrix():
    factor = 0.1
    pts1 = np.float32([
        [0, 0],
        [0, 10],
        [10, 0],
    ])

    pts2 = np.float32([[0, 0]])

    while len(pts2) < 3:
        new_pt = np.float32([random.randint(0, 10) * factor, random.randint(0, 10) * factor])
        where = np.where(pts2 == new_pt)
        if len(where[0]) == 0:
            pts2 = np.append([new_pt], pts2, axis=0)

    pts2 += pts1

    M = cv2.getAffineTransform(pts1, pts2)
    return M


def load_images(path, scale):
    images = []
    list = os.listdir(path)
    for file_name in list:
        file_path = os.path.join(path, file_name)

        img = cv2.imread(file_path)
        if scale != 1:
            img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

        images.append(img)

    return images


def warp_affine(img, M, w, h):
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))

def get_border(img, w, h, type):
    white = np.uint8([255, 255, 255])
    if type == 'minY':
        for y in range(h):
            for x in range(w):
                color = img[y][x]
                if not np.array_equal(color, white):
                    return y
    elif type == 'minX':
        for x in range(w):
            for y in range(h):
                color = img[y][x]
                if not np.array_equal(color, white):
                    return x
    if type == 'maxY':
        for y in range(h):
            for x in range(w):
                color = img[h - 1 - y][x]
                if not np.array_equal(color, white):
                    return h - 1
    if type == 'maxX':
        for y in range(h):
            for x in range(w):
                color = img[y][w - 1 - x]
                if not np.array_equal(color, white):
                    return w - 1

def save_image(filename, img):
    # (h, w) = img.shape[:2]
    # min_x = get_border(img, w, h, 'minX')
    # max_x = get_border(img, w, h, 'maxX')
    # min_y = get_border(img, w, h, 'minY')
    # max_y = get_border(img, w, h, 'maxY')

    cv2.imwrite(filename, img)
    # print(min_x, min_y, max_x, max_y)

def apply_point(M, x, y):
    result = np.dot(M, np.float64([x, y, 1]))
    return np.int32(result)[:2]

def main():
    # scale = 0.25
    all_image = load_images('./data/img/all', 1)
    item_image = load_images('./data/img/items', 1)
    # item_image = list(map(lambda img: cv2.resize(img, dsize=None, fx=scale, fy=scale), origin_item_image))

    for _ in range(1):
        M = get_matrix()
        for all in [all_image[0]]:
            (h, w) = all.shape[:2]
            copy_M = np.append(M, [np.float64([0,0,1])], axis=0)
            (x, y) = apply_point(copy_M, 0, 0)
            (w, h) = apply_point(copy_M, w, h)

            target_img = warp_affine(all, M, w, h)

            for index in range(len(item_image)):
                template = item_image[index]
                h, w = template.shape[:2]  # (rows, cols)

                (w, h) = apply_point(copy_M, w, h)

                template = warp_affine(template, M, w, h)

                # 模板匹配
                res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
                threshold = .8
                loc = np.where(res >= threshold)

                for pt in zip(*loc[::-1]):  # Switch collumns and rows
                    cv2.rectangle(target_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

                path = "./data/tmp/all/{}".format(index)
                os.makedirs(path, exist_ok=True)
                save_image(os.path.join(path, '{}.png'.format(int(round(time.time() * 1000)))), template)


            cv2.imwrite('result.png', target_img)

if __name__ == '__main__':
    main()
