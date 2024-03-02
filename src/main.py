import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pep8

glob_higher = 0


@dataclass
class Line:
    x1: int
    y1: int
    x2: int
    y2: int

    def slope(self):
        if self.x2 - self.x1 == 0:  # Avoid division by zero
            return float("inf")  # Infinite slope
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def length(self):
        return

    def avg_with(self, other):
        if not (isinstance(other, Line)):
            return None
        avx1 = (self.x1 + other.x1) // 2
        avx2 = (self.x2 + other.x2) // 2
        avy1 = (self.y1 + other.y1) // 2
        avy2 = (self.y2 + other.y2) // 2
        return Line(avx1, avy1, avx2, avy2)

    def __str__(self):
        return f"({self.x1},{self.y1})->({self.x2},{self.y2})"


class ColorHSV:
    """class for HSV colors:
    - HSV is kind of abstract when stored as a numpy array
    - just useful for programming purposes.
    """

    def __init__(self, hsv=None):
        if hsv is None:
            self.hsv = np.array([0, 0, 0], dtype=np.uint8)
        else:
            self.hsv = np.array(hsv, dtype=np.uint8)

    def _check_range(self, value, min_val, max_val):
        return max(min_val, min(value, max_val))

    @property
    def hue(self):
        return self.hsv[0]

    @hue.setter
    def hue(self, h):
        h = self._check_range(h, 0, 179)
        self.hsv[0] = h

    @property
    def saturation(self):
        return self.hsv[1]

    @saturation.setter
    def saturation(self, s):
        s = self._check_range(s, 0, 255)
        self.hsv[1] = s

    @property
    def value(self):
        return self.hsv[2]

    @value.setter
    def value(self, v):
        v = self._check_range(v, 0, 255)
        self.hsv[2] = v

    def __str__(self):
        return f"[H={self.hsv[0]}, S={self.hsv[1]}, V={self.hsv[2]}]"

class HsvColorRange:
    """class for HSV color ranges
    - Intended to simplify broadening and minimizing
    HSV color ranges"""

    def __init__(self, name, v1=None, v2=None):
        self.name: str = name
        self.lb = ColorHSV(v1)
        self.ub = ColorHSV(v2)

    def display_colors(self):
        # Create an image for each color
        lower_color_img = np.full((100, 100, 3), self.lb, dtype=np.uint8)
        upper_color_img = np.full((100, 100, 3), self.ub, dtype=np.uint8)

        # Convert from HSV to BGR for display
        lower_color_bgr = cv2.cvtColor(lower_color_img, cv2.COLOR_HSV2BGR)
        upper_color_bgr = cv2.cvtColor(upper_color_img, cv2.COLOR_HSV2BGR)

        # Display the images
        cv2.imshow(f"{self.name} Lower Bound", lower_color_bgr)
        cv2.imshow(f"{self.name} Upper Bound", upper_color_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def modify_bound(self, isLower, h=0, s=0, v=0):
        if isLower:
            self.lb.hue += h
            self.lb.saturation += s
            self.lb.value += v
        else:
            self.ub.hue += h
            self.ub.saturation += s
            self.ub.value += v

    def get_range(self):
        return [self.lb.hsv, self.ub.hsv]

    def __str__(self):
        return f"{self.name} color range:\n\tLower:{self.lb}, Upper:{self.ub}\n"


def remove_background_as_rect(images):
    global glob_higher
    lows = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply threshold (or any other preprocessing steps)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Noise removal
        kernel = np.ones((9, 9), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=12)

        # Background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # rect_bg_mask = find_lowest_point_in_bg_mask(image, sure_bg)
        rows = image.shape[0]
        lowest_point = 0
        for i in range(rows):
            if np.any(sure_bg[i, :]):
                lowest_point = max(lowest_point, i)
        lows.append(lowest_point)
    print(lows)
    lowest_point = int(np.average(lows))
    rect_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    rect_mask[lowest_point + 10:, 0:] = 127
    glob_higher = lowest_point
    return rect_mask


def color_thresholding(image, color_masks, rect_mask):
    # initialize a blank image
    blank = np.zeros(image.shape[:2], dtype=np.uint8)
    # initial image removing the rectangular portion of the background

    r_masked_img = cv2.bitwise_and(image, image, mask=rect_mask)
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(r_masked_img, cv2.COLOR_BGR2HSV)
    # Create masks for yellow and white colors
    combined_mask = blank.copy()
    for color_mask in color_masks:
        print(color_mask)
        mk = cv2.inRange(hsv, color_mask[0], color_mask[1])
        combined_mask = cv2.bitwise_or(combined_mask, mk)
    # Combine the masks to get the regions of yellow and white lines
    result = cv2.bitwise_and(image, image, mask=combined_mask)
    exit()
    return result


def sobel_edge_detection(img):
    image = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    sobelx_b = cv2.Sobel(image[:, :, 0], cv2.CV_8UC1, 1, 0, ksize=9)
    sobelx_g = cv2.Sobel(image[:, :, 1], cv2.CV_8UC1, 1, 0, ksize=9)
    sobelx_r = cv2.Sobel(image[:, :, 2], cv2.CV_8UC1, 1, 0, ksize=9)
    rg = cv2.bitwise_or(sobelx_r, sobelx_g)
    b = cv2.bitwise_or(rg, sobelx_b)
    return b


def region_of_interest(image, mid_pt=None, polygons=None):
    height, width = image.shape[0], image.shape[1]
    bottom_left = (int(width * 0.08), height)
    bottom_right = (int(width * 0.92), height)
    # mid_pt =
    if mid_pt is None:
        mid_pt = (550, 250)

    if not (polygons):
        # TODO: One thing that must be changed are the values in this numpy array, right now they are fixed, but they should be able to acclimate to different videos
        # polygons = np.array([[(100, height), (1300, height), (550, 250)]])
        polygons = np.array([[bottom_left, bottom_right, mid_pt]])
    # polygons variable takes in 3 points, plots them on an image, connects them to make a polygon and fills it in.
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def remove_outlier_lines(center_x, center_y, lines, x_slope_thresh, y_slope_thresh):
    # List to store line segments closest to the center
    closest_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # Extract line coordinates
        # Calculate the midpoint of the line segment
        if abs(x1 - x2) > x_slope_thresh and abs(y1 - y2) > y_slope_thresh:
            midpoint_x, midpoint_y = (x1 + x2) // 2, (y1 + y2) // 2
            # Calculate the Euclidean distance from the midpoint to the center
            distance_to_center = np.sqrt(
                (midpoint_x - center_x) ** 2 + (midpoint_y - center_y) ** 2
            )
            # Set a threshold distance for considering lines
            thresh_min = 50
            threshold_distance = 230  # Adjust as needed
            # Check if the line is close enough to the center
            if thresh_min < distance_to_center < threshold_distance:
                closest_lines.append(line)
    return np.array(closest_lines)


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if (
            slope < 0
        ):  # Consider a threshold close to zero to avoid division by zero error
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = right_line = None

    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)

    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


class PreProcessingMask:
    """A class that performs several functions on the first
    few frames of a video in order to store data about:
        - Where the road generally is in the picture
        - the general orientation of the lane lines
    these will help in making several masks to better evaluate
    our lane finding algorithm. masks such as:
        - masking the entire road
        - masking the left side of the road
        - masking the right side of the road
        - masking certain colors in the image"""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        yellow_r: HsvColorRange = None,
        white_r: HsvColorRange = None,
    ):
        import icecream

        self.first_few_frames = [cap.read()[1] for _ in range(8)]
        self.height = self.first_few_frames[0].shape[0]
        self.width = self.first_few_frames[0].shape[1]
        self.averaged_lines = []
        if yellow_r is None:
            self.yellow_range = HsvColorRange(
                "yellow", np.array([15, 60, 20]), np.array([25, 255, 255])
            )
        else:
            self.yellow_range = yellow_r
        if white_r is None:
            self.white_range = HsvColorRange(
                "white", np.array([0, 0, 180]), np.array([255, 255, 255])
            )
        else:
            self.white_range = white_r

        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.recty = remove_background_as_rect(self.first_few_frames)
        
        self.higher = glob_higher
        self.centroid = [self.width // 2, self.higher]
        self.get_preprocessing_mask()

        # cv2.imshow('test', self.recty)
        # cv2.waitkey(0)
        # cv2.destroyAllWindows()
        # exit()

    def get_color_masks(self):
        """mainly used when passing them in as arguments to
        be used by the color_thresholding function"""
        return [self.yellow_range.get_range(), self.white_range.get_range()]

    # def perspectTrans(self, left, right):
    #     tl = [left.x2, left.y2]
    #     bl = [left.x1, left.y1]
    #     tr = [right.x2, right.y2]
    #     br = [right.x1, right.y1]
    #     gub = self.first_few_frames[0].copy()

    #     pts1 = np.array([tl, bl, tr, br], dtype=np.float32)
    #     pts2 = np.array(
    #         [[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]],
    #         dtype=np.float32,
    #     )
    #     matrix = cv2.getPerspectiveTransform(pts1, pts2)

    #     # print(type(gub))
    #     # print(type(matrix))
    #     transformed_frame = cv2.warpPerspective(gub, matrix, (self.width, self.height))

    #     cv2.imshow("transformedFrame", transformed_frame)
    #     cv2.waitkey(0)
    #     cv2.destroyAllWindows()

    def get_preprocessing_mask(self):
        """initial attempt at determining where the road is."""
        self.averaged_lines = []
        frames = self.first_few_frames.copy()
        for frame in frames:
            thresh_img = color_thresholding(frame, self.get_color_masks(), self.recty)
            sobel_image = sobel_edge_detection(thresh_img)
            cropped_image = region_of_interest(
                sobel_image, (self.width // 2, self.higher)
            )
            lines = cv2.HoughLinesP(cropped_image, 21, np.pi / 180, 20, 9, 2)
            nearest_to_center = remove_outlier_lines(
                self.centroid[0], self.centroid[1], lines, 2, 2
            )
            avgLn = average_slope_intercept(cropped_image, nearest_to_center)
            self.averaged_lines.append(avgLn)

        left_line = Line(*self.averaged_lines[0][0])
        right_line = Line(*self.averaged_lines[0][1])
        # self.perspectTrans(left_line, right_line)
        for i in range(len(self.averaged_lines) - 1):
            left_line = left_line.avg_with(Line(*self.averaged_lines[i + 1][0]))
            right_line = right_line.avg_with(Line(*self.averaged_lines[i + 1][1]))

        self.higher = min(right_line.y2, left_line.y2)
        polygon_points = np.array(
            [
                [0, self.height],
                [self.width, self.height],
                [right_line.x2 + 90, self.higher],
                [left_line.x2 - 90, self.higher],
            ],
            np.int32,
        )
        self.centroid = [np.mean(polygon_points[:, 0]), np.mean(polygon_points[:, 1])]
        cv2.fillPoly(self.mask, [polygon_points], 255)
        blank = np.zeros(self.first_few_frames[0].shape[:2], dtype=np.uint8)

        l_mk = cv2.rectangle(
            blank.copy(),
            (0, self.height // 2),
            (int(self.centroid[0]), self.height),
            255,
            -1,
        )
        r_mk = cv2.rectangle(
            blank.copy(),
            (int(self.centroid[0]), self.height // 2),
            (self.width, self.height),
            255,
            -1,
        )
        self.lane_masks = [
            cv2.bitwise_and(l_mk, self.mask),
            cv2.bitwise_and(r_mk, self.mask),
        ]

        time.sleep(1)
        # self.test_mask_validity()
        self.determine_large_areas()

    def find_straightest_dotted_line(self, frame):
        """Finds the straightest dotted line in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        smoothed_contours = self.image_contouring(frame, contours)

        best_line = None
        max_length = 0

        for contour in smoothed_contours:
            # Filter small contours
            if cv2.contourArea(contour) < 100:  # Threshold might need adjustment
                continue
            # Fit a line to the contour
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            length = np.sqrt(vx**2 + vy**2)
            # Update the best line if this one is straighter
            if length > max_length:
                max_length = length
                best_line = [vx, vy, x, y]
        # Draw the best line
        if best_line is not None:
            vx, vy, x, y = best_line
            # lefty = int((-x*vy/vx) + y)
            # righty = int(((gray.shape[1]-x)*vy/vx)+y)
            return best_line
        return

    def find_largest_solid_line(self, frame):
        """Finds the largest solid line within the largest contour"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        smoothed_contours = self.image_contouring(frame, contours)

        largest_contour = None
        max_area = 0
        # Find the largest contour
        for contour in smoothed_contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour

        # Fit a line to the largest contour
        if largest_contour is not None:
            [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            return [vx, vy, x, y]

    def draw_line(self, frame, line, color):
        vx, vy, x, y = line
        lefty = int((-x * vy / vx) + y)
        righty = int(((self.width - x) * vy / vx) + y)
        cv2.line(frame, (self.width - 1, righty), (0, lefty), color, 2)

    def image_contouring(self, image, contours, alpha=0.01):
        """Smooths the contours of an image."""
        smoothed_contours = []
        for contour in contours:
            # Approximate the contour
            epsilon = alpha * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            smoothed_contours.append(approx)
        return smoothed_contours

    def test_mask_validity(self, frame, mask):
        """Identify whether there are large regions in the image
        after applying the mask
        TODO: Determine whether this function is actually necessary"""
        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and smooth contours
        smoothed_contours = self.image_contouring(frame, contours)

        # cv2.drawContours(frame, [smoothed_contours],-1,(255))
        for contour in contours:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)  # Green contours

        # Display the frame with drawn contours

        return False
        # return large_area_count > 0  # Returns True if large areas are detected

    def calculate_slope(self, line):
        # Assuming line is in format (vx, vy, x, y)
        if line[0] != 0:
            return line[1] / line[0]  # slope = vy / vx
        return float("inf")  # Infinite slope

    def is_within_margin(self, slope, avg_slope, margin):
        """margin of error for comparing solid or dotted yellow lines individually
        against the average slope. (when solid and dotted don't have the same slope)"""
        return abs(slope - avg_slope) <= margin

    def draw_line(self, frame, line, color):
        """draws line to a frame"""
        vx, vy, x, y = line
        lefty = int((-x * vy / vx) + y)
        righty = int(((self.width - x) * vy / vx) + y)
        cv2.line(frame, (self.width - 1, righty), (0, lefty), color, 2)

    def determine_large_areas(self):
        """Originally was going to be used a different way
        it now tries to identify the slopes of the largest solid and
        dotted lines to inform future attempts."""
        frames = self.first_few_frames
        left_lines = []
        right_lines = []
        l_slopes = []
        r_slopes = []
        avg_l_slope = 0
        avg_r_slope = 0
        var_l_slope = 0.2
        var_r_slope = 0.2
        l_margin = 0.1  # Example margin, can be adjusted
        r_margin = 0.1  # Example margin, can be adjusted

        for i, frame in enumerate(frames):
            thresh = color_thresholding(frame, self.get_color_masks(), self.recty)
            fr = cv2.bitwise_and(thresh, thresh, mask=self.mask)
            fr_1 = cv2.bitwise_and(thresh, thresh, mask=self.lane_masks[0])
            fr_r = cv2.bitwise_and(thresh, thresh, mask=self.lane_masks[1])

            solidLnLeft = self.find_largest_solid_line(fr_1.copy())
            solidLnRight = self.find_largest_solid_line(fr_r.copy())
            dotLnLeft = self.find_straightest_dotted_line(fr_1.copy())
            dotLnRight = self.find_straightest_dotted_line(fr_r.copy())
            try:
                if solidLnLeft == dotLnLeft:
                    self.draw_line(fr, solidLnLeft, (255, 0, 0))
                    left_lines.append(solidLnLeft)
                    l_slopes.append(self.calculate_slope(solidLnLeft))
                    avg_l_slope = np.mean(l_slopes)
                    var_l_slope = np.var(l_slopes)
            except:
                pass
            try:
                if solidLnRight == dotLnRight:
                    self.draw_line(fr, solidLnRight, (255, 0, 0))
                    right_lines.append(solidLnRight)
                    r_slopes.append(self.calculate_slope(solidLnRight))
                    avg_r_slope = np.mean(r_slopes)
                    var_r_slope = np.var(r_slopes)
            except:
                pass
            # Process solid and dotted lines for left lane
            if solidLnLeft is not None:
                slope = self.calculate_slope(solidLnLeft)
                if self.is_within_margin(slope, avg_l_slope, l_margin):
                    left_lines.append(solidLnLeft)
                    l_slopes.append(slope)

            if dotLnLeft is not None:
                slope = self.calculate_slope(dotLnLeft)
                if self.is_within_margin(slope, avg_l_slope, l_margin):
                    left_lines.append(dotLnLeft)
                    l_slopes.append(slope)

            # Process solid and dotted lines for right lane
            if solidLnRight is not None:
                slope = self.calculate_slope(solidLnRight)
                if self.is_within_margin(slope, avg_r_slope, r_margin):
                    right_lines.append(solidLnRight)
                    r_slopes.append(slope)

            if dotLnRight is not None:
                slope = self.calculate_slope(dotLnRight)
                if self.is_within_margin(slope, avg_r_slope, r_margin):
                    right_lines.append(dotLnRight)
                    r_slopes.append(slope)

            # Update average and variance of slopes
            if l_slopes:
                avg_l_slope = np.mean(l_slopes)
                var_l_slope = np.var(l_slopes)
                if var_l_slope < l_margin:
                    l_margin = var_l_slope

            if r_slopes:
                avg_r_slope = np.mean(r_slopes)
                var_r_slope = np.var(r_slopes)
                if var_r_slope < l_margin:
                    r_margin = var_r_slope

        self.l_slope = avg_l_slope
        self.r_slope = avg_r_slope
        self.draw_triangle()

    def draw_triangle(self):
        """Adds cuts out little mask from the  bottom center of image"""
        mk = np.zeros((self.height, self.width), dtype=np.uint8)
        # Calculate end points for the left and right lines
        if self.l_slope != float("inf") and self.l_slope != -float("inf"):
            x_left = int(
                self.centroid[0] - (self.centroid[1] - self.height) / self.l_slope
            )
        else:
            x_left = self.centroid[0]

        if self.r_slope != float("inf") and self.r_slope != -float("inf"):
            x_right = int(
                self.centroid[0] + (self.height - self.centroid[1]) / self.r_slope
            )
        else:
            x_right = self.centroid[0]

        polygon_points = np.array(
            [
                (x_left, self.height),
                (x_right, self.height),
                (self.centroid[0], self.centroid[1]),
            ],
            np.int32,
        )

        cv2.fillPoly(mk, [polygon_points], 255)
        mk = cv2.bitwise_not(mk)
        self.mask = cv2.bitwise_and(self.mask, self.mask, mask=mk)
        self.lane_masks[0] = cv2.bitwise_and(
            self.lane_masks[0], self.lane_masks[0], mask=mk
        )
        self.lane_masks[1] = cv2.bitwise_and(
            self.lane_masks[1], self.lane_masks[1], mask=mk
        )


def avg_lin(lines):
    average = None
    if lines is not None and len(lines) > 0:
        # Separate out x and y coordinates
        x1s, y1s, x2s, y2s = zip(*[line[0] for line in lines])

        # Calculate the mean for each coordinate
        avg_x1 = int(np.mean(x1s))
        avg_y1 = int(np.mean(y1s))
        avg_x2 = int(np.mean(x2s))
        avg_y2 = int(np.mean(y2s))

        average = np.array([avg_x1, avg_y1, avg_x2, avg_y2])
    return average


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line  # Extract the coordinates from the array
                cv2.line(
                    line_image, (x1, y1), (x2, y2), (255, 0, 0), 10
                )  # Increase thickness here if needed
    return line_image


def calculate_line_slope(line):
    """Calculate the slope of a given line"""
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:  # Avoid division by zero
        return float("inf")  # Infinite slope
    return (y2 - y1) / (x2 - x1)


def filter_lines_by_slope(lines, avg_slope, tolerance=0.1):
    """Filter lines based on their slope, preserving the original format"""
    if lines is None:
        return lines  # Return None if no lines are detected

    filtered_lines = []
    for line in lines:
        slope = calculate_line_slope(line[0])
        if abs(slope - avg_slope) <= tolerance:  # Check if slope is within tolerance
            filtered_lines.append(line)

    return np.array(filtered_lines) if filtered_lines else lines


def darken_outside_mask(image, mask):
    # Darken the entire image
    darkened_image = cv2.addWeighted(image, 0.5, np.zeros_like(image), 0.5, 0)

    # Keep only the region inside the mask from the original image
    masked_original = cv2.bitwise_and(image, image, mask=mask)

    # Invert the mask to darken the region outside the mask
    inverted_mask = cv2.bitwise_not(mask)

    # Keep only the region outside the mask from the darkened image
    masked_darkened = cv2.bitwise_and(darkened_image, darkened_image, mask = inverted_mask)
    
    # Combine the two images
    combined = cv2.add(masked_original, masked_darkened)
    return combined


def run(name, prp_mask):
    cap = cv2.VideoCapture(name)
    min_len = prp_mask.height - prp_mask.higher - 30
    i = 0
    avg_left = None
    avg_right = None

    while cap.isOpened():
        frame = cap.read()[1]

        if frame is None:
            break
        thresh_img = color_thresholding(
            frame, prp_mask.get_color_masks(), prp_mask.recty
        )
        kernel = np.ones((5, 5), np.uint8)
        new = cv2.dilate(thresh_img, kernel)
        new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        # new = sobel_edge_detection(thresh_img)
        gang = cv2.bitwise_and(new, new, mask=prp_mask.mask)
        gang = cv2.cvtColor(gang, cv2.COLOR_GRAY2RGB)
        left_img = cv2.bitwise_and(new, new, mask=prp_mask.lane_masks[0])
        right_img = cv2.bitwise_and(new, new, mask=prp_mask.lane_masks[1])
        givenMask = cv2.bitwise_or(prp_mask.lane_masks[0], prp_mask.lane_masks[1])
        # dumbfumb = cv2.bitwise_and()

        # # focus region to be the lower triangular portion of the image
        l_lines = cv2.HoughLinesP(
            image=left_img,
            rho=9,
            theta=np.pi / 180,
            threshold=50,
            lines=np.array([]),
            minLineLength=min_len,
            maxLineGap=min_len * 0.7,
        )
        filtered_l = filter_lines_by_slope(l_lines, prp_mask.l_slope, 0.25)
        average_left = avg_lin(filtered_l)

        r_lines = cv2.HoughLinesP(
            image=right_img,
            rho=9,
            theta=np.pi / 180,
            threshold=50,
            lines=np.array([]),
            minLineLength=min_len,
            maxLineGap=min_len * 0.7,
        )
        filtered_r = filter_lines_by_slope(r_lines, prp_mask.r_slope, 0.25)
        average_right = avg_lin(filtered_r)

        if i % 70 == 0:
            try:
                if avg_left is None:
                    avg_left = average_left
                else:
                    avg_left = avg_lin(np.array([avg_left, average_left]))
                if avg_right is None:
                    avg_right = average_right
                else:
                    avg_left = avg_lin(np.array(avg_right, average_right))
            except:
                pass
        i += 1

        if average_left is None:
            average_left = avg_left
        if average_right is None:
            average_right = avg_right

        averaged_lines = [average_left, average_right]

        # line_image = display_lines(frame, averaged_lines)
        # combo_image = cv2.addWeighted(frame, 0.75, line_image, 1, 1)
        # goob = cv2.bitwise_and(frame, frame, mask=givenMask)
        # goob = darken_outside_mask(frame, givenMask)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.75, line_image, 1, 1)

        # line_image = display_lines(gang, averaged_lines)
        # combo_image = cv2.addWeighted(gang, 0.75, line_image, 1, 1)

        try:
            prp_mask.l_slope = np.mean(
                [prp_mask.l_slope, calculate_line_slope(average_left)]
            )
            prp_mask.r_slope = np.mean(
                [prp_mask.r_slope, calculate_line_slope(average_right)]
            )
        except:
            pass
        cv2.imshow("result", combo_image)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    cwd = os.getcwd()
    data_dir = os.listdir(os.path.join(cwd, "data"))
    videos = [os.path.join("data", file) for file in data_dir]

    for v in videos:
        # try:
        c = cv2.VideoCapture(v)
        prp_mask = PreProcessingMask(c)
        run(v, prp_mask)
        # except:
        #     print(f"Error occured for {v}")