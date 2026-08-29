import numpy as np
import cv2
from pathlib import Path
import time

# print(cv2.getBuildInformation())
# print(Path(cv2.__file__).parent / "qt")
# img = np.zeros((300,300,3), dtype=np.uint8)
# img[:] = (255,0,0)

# img = cv2.imread("/home/rajatnai49/Pictures/HIiC0FYXsAALIdo.jpeg")

# if img is None:
# raise FileNotFoundError("Not be able to load image")

# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Image", 800, 600)
# print(img[200,200])

# print(img.shape)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
counter = 0
start_time = time.time()

cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cam", 800, 600)
cv2.namedWindow("Cam2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cam2", 800, 600)
cv2.namedWindow("Cam3", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cam3", 800, 600)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    counter += 1
    elapsed = int((time.time() - start_time))

    fps = 0

    if elapsed > 0:
        fps = int(counter / elapsed)

    # cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    # cv2.putText(frame, f"FPS: {fps:.1f}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Cam", frame)
    cv2.imshow("Cam2", rgb)
    cv2.imshow("Cam3", grey)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# img = np.zeros((300, 600, 3), dtype=np.uint8)

# cv2.putText(
#     img, "Hello from Rajat", (150, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
# )

# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#####

# img = cv2.imread("/home/rajatnai49/Downloads/COLE.jpg")

# if img is None:
#     raise FileNotFoundError("Failed to load image")

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# def mouse_callback(event, x, y, flags, params):
#     if event == cv2.EVENT_MBUTTONDOWN:
#         print(f"x={x}, y={y}")

# edges = cv2.Canny(
#     gray,
#     100,
#     200
# )

# _, thresh = cv2.threshold(
#     gray,
#     120,
#     255,
#     cv2.THRESH_BINARY
# )

# contours, _ = cv2.findContours(
#     edges,
#     cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE
# )

# cv2.drawContours(
#     img,
#     contours,
#     -1,
#     (0,255,0),
#     2
# )

# cv2.imshow("Normal", img)
# cv2.imshow("Gray", gray)
# cv2.imshow("Thres", thresh)
# cv2.imshow("Edges", edges)

# while True:
#     key = cv2.waitKey(1)

#     if key == ord('q'):
#         break

# cv2.destroyAllWindows()
