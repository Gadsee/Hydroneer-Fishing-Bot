import time

import cv2
import mss
import numpy
import pyautogui
from numpy import ndarray

LOWER_HSV_RED_RANGE_LOWER = numpy.array([0, 70, 70])
LOWER_HSV_RED_RANGE_UPPER = numpy.array([10, 255, 255])

UPPER_HSV_RED_RANGE_LOWER = numpy.array([160, 70, 70])
UPPER_HSV_RED_RANGE_UPPER = numpy.array([180, 255, 255])

MIN_PIXEL_MOVEMENT_DOWN = -19
MAX_WAIT_TIME = 16


def track_and_check_movement(
    pixels: ndarray, prev_y_coordinates: list[int], start_time: float
) -> tuple[ndarray, list[int], float]:
    hsv_pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)

    # Since HSV has the color red on both of its ends, we need two masks
    lower_mask = cv2.inRange(
        hsv_pixels, LOWER_HSV_RED_RANGE_LOWER, LOWER_HSV_RED_RANGE_UPPER
    )
    upper_mask = cv2.inRange(
        hsv_pixels, UPPER_HSV_RED_RANGE_LOWER, UPPER_HSV_RED_RANGE_UPPER
    )
    mask = lower_mask + upper_mask

    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, width, height = cv2.boundingRect(largest_contour)
        cv2.rectangle(pixels, (x, y), (x + width, y + height), (0, 255, 0), 2)

        centroid_y = y + int(height / 2)

        prev_y_coordinates.append(centroid_y)
        elapsed_time = time.time() - start_time

        if elapsed_time >= 4:
            average_y = sum(prev_y_coordinates) / len(prev_y_coordinates)
            if average_y - centroid_y < MIN_PIXEL_MOVEMENT_DOWN:
                pyautogui.click()
                time.sleep(0.25)
                pyautogui.click()
                time.sleep(1.25)
                prev_y_coordinates.clear()
                start_time = time.time()

        # If it took too long to wait for a fish, re-try
        if elapsed_time >= MAX_WAIT_TIME:
            pyautogui.click()
            time.sleep(0.25)
            pyautogui.click()
            time.sleep(1.25)
            prev_y_coordinates.clear()
            start_time = time.time()

    return pixels, prev_y_coordinates, start_time


def main() -> None:
    prev_y_coordinates = []
    start_time = time.time()
    pyautogui.click()

    while True:
        with mss.mss() as screenshot_taker:
            screenshot = screenshot_taker.grab((960, 100, 1420, 980))

        pixels = numpy.array(screenshot.pixels)
        (
            tracked_frame,
            prev_y_coordinates,
            start_time,
        ) = track_and_check_movement(pixels, prev_y_coordinates, start_time)

        cv2.imshow("Hydroneer Fish Bot Vision", tracked_frame)
        cv2.setWindowProperty(
            "Hydroneer Fish Bot Vision", cv2.WND_PROP_TOPMOST, 1
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
