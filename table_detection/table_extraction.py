import cv2
import numpy as np

from statistics import median
from utils.image_util import grayscale

try:
    from PIL import Image
except ImportError:
    import Image


class TableExtraction:

    def __sort_contours(self, cnts):
        reverse = False
        i = 1
        bounding_boxes = [cv2.boundingRect(c) for c in cnts]
        cnts, bounding_boxes = zip(*sorted(zip(cnts, bounding_boxes),
                                           key=lambda b: b[1][i],
                                           reverse=reverse))

        _, _, widths, heights = zip(*bounding_boxes)
        mean_w, mean_h = int(median(widths)), int(median(heights))
        w_diff, h_diff = mean_w * 0.15, mean_h * 0.15
        w_lower, w_upper = mean_w - w_diff, mean_w + w_diff
        h_lower, h_upper = mean_h - h_diff, mean_h + h_diff

        cells = [[x, y, w, h] for x, y, w, h in bounding_boxes if
                 w_lower < w < w_upper and h_lower < h < h_upper]
        return cnts, cells

    def __create_kernels(self, image):
        kernel_len_ver = 70
        kernel_len_hor = 30
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               (1, kernel_len_ver))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               (kernel_len_hor, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return ver_kernel, hor_kernel, kernel

    def __get_vertical_lines(self, image_bin, ver_kernel):
        image = cv2.erode(image_bin, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(image, ver_kernel, iterations=3)
        return vertical_lines

    def __get_horizontal_lines(self, image_bin, hor_kernel):
        image = cv2.erode(image_bin, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(image, hor_kernel, iterations=3)
        return horizontal_lines

    def __sort_cells_to_rows_and_columns(self, cells, mean):
        rows, column = [], []
        column.append(cells[0])
        previous = cells[0]
        for i in range(1, len(cells)):
            if cells[i][1] <= previous[1] + mean / 2:
                column.append(cells[i])
                previous = cells[i]
                if i == len(cells) - 1:
                    rows.append(column)
            else:
                rows.append(column)
                column = []
                previous = cells[i]
                column.append(cells[i])
        return rows

    def __calculate_maximum_number_of_cells(self, rows):
        countcol = 0
        for i in range(len(rows)):
            countcol = len(rows[i])
            if countcol > countcol:
                countcol = countcol
        return countcol, len(rows) - 1

    def __get_column_centers(self, rows, i):
        centers = [int(rows[i][j][0] + rows[i][j][2] / 2) for j in
                   range(len(rows[i])) if rows[0]]
        centers = np.array(centers)
        centers.sort()
        return centers

    def __get_final_cells(self, rows, countcol, centers):
        final_cells = []
        for i in range(len(rows)):
            lis = []
            for k in range(countcol):
                lis.append([])
            for j in range(len(rows[i])):
                diff = abs(centers - (rows[i][j][0] + rows[i][j][2] / 4))
                minimum = min(diff)
                indexing = list(diff).index(minimum)
                lis[indexing].append(rows[i][j])
            final_cells.append(lis)
        return final_cells

    def extract_table(self, image):
        gray = grayscale(image)
        thresh, image_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        image_bin = ~image_bin

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel,
                                     iterations=2)

        ver_kernel, hor_kernel, kernel = self.__create_kernels(image)
        vertical_lines = self.__get_vertical_lines(image_bin, ver_kernel)
        horizontal_lines = self.__get_horizontal_lines(image_bin, hor_kernel)
        image_vh = cv2.addWeighted(vertical_lines, 1, horizontal_lines, 1, 0.)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image_vh = cv2.dilate(image_vh, kernel, iterations=2)
        image_vh = ~image_vh

        contours, hierarchy = cv2.findContours(image_vh, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        contours, bounding_boxes = self.__sort_contours(contours)

        heights = [bounding_boxes[i][3] for i in range(len(bounding_boxes))]
        mean = np.mean(heights)

        rows = self.__sort_cells_to_rows_and_columns(bounding_boxes, mean)
        countcol, i = self.__calculate_maximum_number_of_cells(rows)

        center = self.__get_column_centers(rows, i)
        final_cells = self.__get_final_cells(rows, countcol, center)

        return final_cells
