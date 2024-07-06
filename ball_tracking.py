import cv2
import numpy as np

color_ranges = {
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'green': [(36, 50, 50), (86, 255, 255)],
    'white': [(0, 0, 220), (180, 30, 255)],
    'orange': [(5, 150, 150), (15, 255, 255)]
}

video_path = 'C:\\Users\\Nayeem\\Downloads\\AI Assignment video.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

quadrants = {
    1: [(0, 0), (frame_width // 2, frame_height // 2)],
    2: [(frame_width // 2, 0), (frame_width, frame_height // 2)],
    3: [(0, frame_height // 2), (frame_width // 2, frame_height)],
    4: [(frame_width // 2, frame_height // 2), (frame_width, frame_height)]
}

def get_quadrant(x, y):
    for q, ((x1, y1), (x2, y2)) in quadrants.items():
        if x1 <= x < x2 and y1 <= y < y2:
            return q
    return None

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

tracking_data = []

frame_count = 0
ball_positions = {color: None for color in color_ranges}
ball_last_quadrant = {color: None for color in color_ranges}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_time = frame_count / fps

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:
                adjusted_radius = radius + 5 if color == 'white' else radius  # Adjust radius for white ball
                cv2.circle(frame, (int(x), int(y)), int(adjusted_radius), (0, 255, 255), 2)
                cv2.putText(frame, color, (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                current_quadrant = get_quadrant(int(x), int(y))
                if ball_positions[color] is None:
                    ball_positions[color] = (int(x), int(y))

                last_quadrant = ball_last_quadrant[color]
                if last_quadrant is not None and current_quadrant != last_quadrant:
                    tracking_data.append((frame_time, last_quadrant, color, 'Exit'))
                    cv2.putText(frame, f"Exit - {color} - Q{last_quadrant} - {frame_time:.2f}s",
                                (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    if current_quadrant is not None:
                        tracking_data.append((frame_time, current_quadrant, color, 'Entry'))
                        cv2.putText(frame, f"Entry - {color} - Q{current_quadrant} - {frame_time:.2f}s",
                                    (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    ball_last_quadrant[color] = current_quadrant
                elif last_quadrant is None and current_quadrant is not None:
                    tracking_data.append((frame_time, current_quadrant, color, 'Entry'))
                    cv2.putText(frame, f"Entry - {color} - Q{current_quadrant} - {frame_time:.2f}s",
                                (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    ball_last_quadrant[color] = current_quadrant
                ball_positions[color] = (int(x), int(y))
            else:
                ball_positions[color] = None
                ball_last_quadrant[color] = None
        else:
            ball_positions[color] = None
            ball_last_quadrant[color] = None

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

with open('tracking_data.txt', 'w') as f:
    for entry in tracking_data:
        f.write(', '.join(map(str, entry)) + '\n')

print("Processing complete. Output video and tracking data saved.")
