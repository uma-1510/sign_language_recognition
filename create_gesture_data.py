import cv2
import numpy as np
import os

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)

# Classes
classes = range(1, 11)  # 1 to 10

# Number of images per class
num_train_images = 300
num_test_images = 50

# Create folders if not exist
for cls in classes:
    os.makedirs(f"gesture/train/{cls}", exist_ok=True)
    os.makedirs(f"gesture/test/{cls}", exist_ok=True)

cam = cv2.VideoCapture(0)

for element in classes:
    print(f"Starting capture for class {element}...")
    num_frames = 0
    num_imgs_taken = 0
    background = None  # reset background for each class

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            continue

        frame_copy = frame.copy()
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        if num_frames < 60:
            cal_accum_avg(gray_frame, accumulated_weight)
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else:
            hand = segment_hand(gray_frame)
            if hand is not None:
                thresholded, hand_segment = hand
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255,0,0), 1)
                cv2.putText(frame_copy, f"Images taken: {num_imgs_taken} / {num_test_images}", 
                            (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Thresholded Hand Image", thresholded)

                # Save to train or test
                if num_imgs_taken < num_train_images:
                    path = f"gesture/train/{element}/{num_imgs_taken}.jpg"
                else:
                    if num_imgs_taken < num_train_images + num_test_images:
                        path = f"gesture/test/{element}/{num_imgs_taken - num_train_images}.jpg"
                    else:
                        break  # done with this class

                cv2.imwrite(path, thresholded)
                num_imgs_taken += 1

        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
        cv2.imshow("Sign Detection", frame_copy)
        num_frames += 1

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC to exit
            break

cam.release()
cv2.destroyAllWindows()
print("Dataset capture complete!")
