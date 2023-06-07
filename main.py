from PIL import ImageGrab
import numpy as np
import cv2
import pyautogui

pyautogui.screenshot().save('screenshot.png')

def detect_bricks():
    background_color = np.array([48, 128, 72], dtype=np.uint8)
    brick_color = np.array([255, 255, 255], dtype=np.uint8)

    background_mask = cv2.inRange(image, background_color, background_color)
    brick_mask = cv2.inRange(image, brick_color, brick_color)

    contours, _ = cv2.findContours(brick_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area_threshold = 1000
    min_white_percentage = 50.0

    filtered_bricks = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        contour_mask = np.zeros_like(brick_mask)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)

        white_pixels = cv2.countNonZero(cv2.bitwise_and(contour_mask, brick_mask))
        white_area_percentage = (white_pixels / area) * 100.0

        if area > min_area_threshold and white_area_percentage >= min_white_percentage:
            filtered_bricks.append((x, y, w, h, white_area_percentage))

    result = image.copy()

    # Create a binary mask for the bricks
    brick_binary = np.zeros_like(brick_mask)
    for brick_rect in filtered_bricks:
        x, y, w, h, white_percentage = brick_rect
        cv2.rectangle(brick_binary, (x, y), (x + w, y + h), 255, -1)

    # Find connected components of the brick mask
    _, labels = cv2.connectedComponents(brick_binary)

    # Iterate through each brick
    for brick_rect, label in zip(filtered_bricks, range(1, len(filtered_bricks) + 1)):
        x, y, w, h, white_percentage = brick_rect

        # Check if the brick is connected to any other brick
        is_connected = (
            (y > 0 and labels[y - 1, x + (w // 2)] != label) or  # Check above
            (y + h < image.shape[0] and labels[y + h, x + (w // 2)] != label) or  # Check below
            (x > 0 and labels[y + (h // 2), x - 1] != label) or  # Check left
            (x + w < image.shape[1] and labels[y + (h // 2), x + w] != label)  # Check right
        )

        # Draw the entire brick rectangle in red
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw only the connecting sides in green if the brick is connected to another brick
        

    cv2.imshow('Filtered Bricks', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_color_rectangles(image, lower_color, upper_color):
    # Convert image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define color range to mask
    mask = cv2.inRange(image_rgb, lower_color, upper_color)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find rectangles around each contour
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append((x, y, w, h))

    return rectangles

# Example usage
if __name__ == '__main__':
    # Load image
    image = cv2.imread('screenshot.png')

    # Define color range (in RGB)
    lower_color = np.array([48, 128, 72])  # Lower range of color (e.g., blue)
    upper_color = np.array([48, 128, 72])  # Upper range of color (e.g., blue)

    # Find rectangles of color
    rectangles = find_color_rectangles(image, lower_color, upper_color)

    if rectangles:
        for rect in rectangles:
            x, y, w, h = rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image = image[y:y+h, x:x+w]
            
        detect_bricks()    
    else:
        print("Color not found in the image.")