import cv2
import numpy
from sklearn.metrics import mean_absolute_error


def check_for_collision(left_edge, right_edge, balls):
    collisions_counter = 0
    for ball in balls:
        if ball[0] - left_edge[0] <= 20:
            collisions_counter += 1
        if right_edge[0] - ball[0] <= 20:
            collisions_counter += 1

    return collisions_counter


def find_balls(frame):
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    ret, binary_frame = cv2.threshold(grey_frame, 150, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    result = []
    for contour in contours:
        (x, y), r = cv2.minEnclosingCircle(contour)
        if 4.2 > r > 3.5:
            result.append([x, y])
    return result


def find_edges(frame):
    edges = cv2.Canny(frame, 100, 200)

    lines = cv2.HoughLinesP(image=edges, rho=1, theta=numpy.pi/180, threshold=10, lines=numpy.array([]),
                            minLineLength=385, maxLineGap=45)
    left_edge = [700, 0, 700, 0]
    right_edge = [240, 0, 240, 0]

    if lines.any():
        for i in range(len(lines)):
            line = lines[i][0]
            x1, y1, = line[0], line[1]
            x2, y2, = line[2], line[3]

            if x1 == x2:
                if x1 < left_edge[0]:
                    left_edge = [x1, y1, x2, y2]
                if x1 > right_edge[0]:
                    right_edge = [x1, y1, x2, y2]

    return [left_edge, right_edge]


def video_processing(file_path):

    cap = cv2.VideoCapture(file_path)
    current_frame_no = 1
    last_collision_frame_no = -1
    collision_counter = 0

    ret_value, frame = cap.read()
    if not ret_value:
        return
    edges = find_edges(frame)
    left_edge, right_edge = edges[0], edges[1]

    balls = find_balls(frame)
    collisions_detected = check_for_collision(left_edge, right_edge, balls)
    if collisions_detected != 0:
        collision_counter += collisions_detected
        last_collision_frame_no = current_frame_no

    while True:
        current_frame_no += 1
        ret_value, frame = cap.read()
        if not ret_value:
            break

        balls = find_balls(frame)
        collisions_detected = check_for_collision(left_edge, right_edge, balls)
        if collisions_detected != 0 and current_frame_no - last_collision_frame_no > 2:
            collision_counter += collisions_detected
            last_collision_frame_no = current_frame_no

    return collision_counter


if __name__ == '__main__':

    correct_results = {}
    calculated_results = {}

    # open data/res.txt and read correct data
    file_results = open("data/res.txt", "r")
    for line in file_results:
        tokens = line.strip().split(",")
        if tokens[0] == "file":
            continue
        correct_results[tokens[0]] = int(tokens[1])

    # iterate through mp4 files and count collisions
    for video in correct_results.keys():
        print(video)
        print("[CORRECT] : " + str(correct_results[video]) + " hits")
        calculated_result = video_processing("data/" + video)
        calculated_results[video] = calculated_result
        print("[CALCULATED] : " + str(calculated_result) + " hits\n")

    correct_numbers_list = list(correct_results.values())
    calculated_numbers_list = list(calculated_results.values())

    mae = mean_absolute_error(correct_numbers_list, calculated_numbers_list)
    print("MAE: " + str(mae))
