import cv2
import numpy as np


def background_removal(img):
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur
    smooth = cv2.GaussianBlur(gray, (95, 95), 0)
    # divide gray by morphology image
    division = cv2.divide(gray, smooth, scale=220)
    img = cv2.cvtColor(division, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    # Edge detection
    edges = cv2.Canny(gray, 10, 90)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    # Find contours in edges, sort by area
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        contour_info.append(
            (
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            )
        )

    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    # max_contour = contour_info[0]
    # Create empty mask and flood fill
    mask = np.zeros(edges.shape)

    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], 255)

    # Smooth mask and blur it
    mask = cv2.dilate(mask, None, iterations=10)
    mask = cv2.erode(mask, None, iterations=10)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    # Create 3-channel alpha mask
    mask_stack = np.dstack([mask] * 3)
    # Blend mask and foreground image
    mask_stack = mask_stack.astype("float32") / 255.0
    img = img.astype("float32") / 255.0
    masked = (mask_stack * img) + ((1 - mask_stack) * (1.0, 1.0, 1.0))
    masked = (masked * 255).astype("uint8")
    # Make the background transparent by adding 4th alpha channel
    tmp = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(masked)
    rgba = [b, g, r, alpha]

    return cv2.merge(rgba, 4)


# cv2.imwrite("shadow_out.png", background_removal(cv2.imread("shadow.png")))
