import cv2
import numpy as np
from ultralytics import YOLO
import os


# Function to count scores based on target points and radii
def count_scores(center, points, radii):
    scores = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1]  # Example scores for each radius
    total_score = 0
    point_scores = []

    for point in points:
        x1, y1, x2, y2, _, _ = [int(i) for i in point]
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        distance = np.sqrt((cx - center[0]) ** 2 + (cy - center[1]) ** 2)

        for i, radius in enumerate(radii):
            if distance <= radius:
                point_scores.append((cx, cy, scores[i]))
                total_score += scores[i]
                break

    return total_score, point_scores


# Function to process and display a single image
def process_image(image_path, model_target_bbox):

    # Reading the target image
    image = cv2.imread(image_path)
    cv2.imshow("input", image)
    cv2.imwrite("input.jpg", image)
    cv2.waitKey(0)
    result = model_target_bbox(image)
    extracted_coor = result[0].boxes.xyxy.tolist()

    # If the target box is completely detected
    if len(extracted_coor) >= 1:

        # Extracted the area
        extracted_coor = extracted_coor[0]
        x1, y1, x2, y2 = [int(i) for i in extracted_coor]
        extracted_area = image[y1:y2, x1:x2]

        # Resize the extracted area
        extracted_area = cv2.resize(extracted_area, (600, 600))

        cv2.imshow("extracted_area", extracted_area)
        cv2.imwrite("extracted_area.jpg", extracted_area)
        cv2.waitKey(0)

        # Making the hsv of the extracted area
        hsv = cv2.cvtColor(extracted_area, cv2.COLOR_BGR2HSV)

        cv2.imshow("hsv", hsv)
        cv2.imwrite("hsv_input.jpg", hsv)
        cv2.waitKey(0)

        lower_yellow = np.array([20, 210, 100])  # Lower boundary of yellow (H, S, V)
        upper_yellow = np.array([30, 255, 255])  # Upper boundary of yellow (H, S, V)

        # Extracting the yellow color
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.erode(mask, (5,5), iterations=6)

        # # Extracting its mask
        # region = cv2.bitwise_and(extracted_area, extracted_area, mask=mask)

        # Extracting the contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Image dimensions and center
        height, width = extracted_area.shape[:2]
        image_center = (width // 2, height // 2)

        # List to store contours and their center distances
        contour_distances = []

        # Calculate the centroid of each contour and its distance to the image center
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distance = np.sqrt((cx - image_center[0]) ** 2 + (cy - image_center[1]) ** 2)
                contour_distances.append((contour, distance))

        # Sort contours by distance (closest first)
        contour_distances.sort(key=lambda x: x[1])

        print([i[1] for i in contour_distances])

        # Eliminating the contour whose distances are greater then 100
        contour_distances = [i for i in contour_distances if i[1] <= 100]

        if len(contour_distances) != 0:

            # Select the closest three contours
            contours = [cd[0] for cd in contour_distances]

            # Combine all contour points into a single array
            all_points = np.vstack([contours[i] for i in range(len(contours))])

            # Calculate the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(all_points)
            center = (int(x), int(y))
            radius = int(radius)

            # Draw the enclosing circle
            cv2.circle(extracted_area, center, radius, (0, 255, 0), 2)
            cv2.circle(extracted_area, center, 5, (255, 0, 0), -1)

            cv2.imshow("center circle", extracted_area)
            cv2.imwrite("center circle.jpg", extracted_area)
            cv2.waitKey(0)

            # Make the prediction from the target and center points
            result = model_target_points(extracted_area, iou=0.3, conf=0.2)

            # Display all the radii
            radii = [16, 28, 56, 85, 117, 142, 172, 200, 230, 260, 290]
            for radius in radii:
                cv2.circle(extracted_area, center, radius, (255, 0, 0), 2)

            cv2.imshow("target_rings", extracted_area)
            cv2.imwrite("target_rings.jpg", extracted_area)
            cv2.waitKey(0)

            # Displaying all the target points
            points = [i for i in result[0].boxes.data.tolist() if int(i[-1]) == 1 and i[-2] > 0.3]
            for point in points:
                x1, y1, x2, y2, _, _ = [int(i) for i in point]
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
                cv2.circle(extracted_area, (cx, cy), 2, (0, 255, 0), -1)

            cv2.imshow("target_points", extracted_area)
            cv2.imwrite("target_points.jpg", extracted_area)
            cv2.waitKey(0)

            # Count the scores and get the point scores
            total_score, point_scores = count_scores(center, points, radii)
            print("Total Score:", total_score)

            # Display the scores on the image
            for (cx, cy, score) in point_scores:
                cv2.putText(extracted_area, str(score), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("target_scores", extracted_area)
            cv2.imwrite("target_scores.jpg", extracted_area)
            cv2.waitKey(0)

            # Display total score on the image
            cv2.putText(extracted_area, f"Total Score: {total_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Target Area", extracted_area)
            cv2.imshow("region", mask)
            cv2.imwrite("mask.jpg", mask)
            cv2.imwrite("result.jpg", extracted_area)
            cv2.waitKey()
            cv2.destroyAllWindows()


# Loading both models
model_target_bbox = YOLO("Models/target_bbox.pt")
model_target_points = YOLO("Models/target_and centers.pt")

# Directory containing the images
image_directory = "targetPointsDataset/train/images"

# Iterate over all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        print(filename)
        image_path = os.path.join(image_directory, filename)
        process_image(image_path, model_target_bbox)
        break