import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load map and video
map_img = cv2.imread('route29_map.png')
map_h, map_w = map_img.shape[:2]
video_path = 'route29_run.mp4'
cap = cv2.VideoCapture(video_path)

# Video frame size (180x120)
video_frame_width = 180
video_frame_height = 120

# Scaling factors to map video frame coordinates to the map coordinates
scale_x = map_w / video_frame_width  # 1200 / 180 = 6.67
scale_y = map_h / video_frame_height  # 424 / 120 = 3.53

# Step 1: Extract player template from a known frame (frame where player is visible)
frame_number = 50  # Example frame number where the player is visible
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Couldn't read the reference frame from the video")

# Player sprite is centered: adjust based on actual sprite size (assumed 16x16)
player_size = 16
x = video_frame_width // 2 - player_size // 2
y = video_frame_height // 2 - player_size // 2

# Crop player sprite
player_template = frame[y:y+player_size, x:x+player_size]

# Convert template to grayscale and equalize histogram
gray_template = cv2.cvtColor(player_template, cv2.COLOR_BGR2GRAY)
gray_template = cv2.equalizeHist(gray_template)

# Step 2: Process video and match template
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
heatmap = np.zeros((map_h, map_w), dtype=np.float32)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (video_frame_width, video_frame_height))  # Resize to match video resolution
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)

    # Template matching
    res = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val > 0.5:  # Confidence threshold
        # Calculate the corresponding location on the full map
        cx = int((max_loc[0] + player_size // 2) * scale_x)
        cy = int((max_loc[1] + player_size // 2) * scale_y)

        # Ensure we are within bounds of the map to avoid errors when cropping
        map_x1 = max(0, cx - video_frame_width // 2)
        map_y1 = max(0, cy - video_frame_height // 2)
        map_x2 = min(map_w, map_x1 + video_frame_width)
        map_y2 = min(map_h, map_y1 + video_frame_height)

        # Extract the region from the map that corresponds to the video frame
        map_section = map_img[map_y1:map_y2, map_x1:map_x2]

        # Update the heatmap based on the player's position on the map
        if 0 <= cx < map_w and 0 <= cy < map_h:
            heatmap[cy, cx] += 1

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()

# Step 3: Normalize and create heatmap
heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap_uint8 = heatmap_norm.astype(np.uint8)
colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(map_img, 0.6, colored_heatmap, 0.4, 0)

# Step 4: Plot Before and After
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot original map (before)
axes[0].imshow(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB))
axes[0].axis("off")
axes[0].set_title("Original Map")

# Plot map with heatmap (after)
axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[1].axis("off")
axes[1].set_title("Map with Heatmap")

# Show the comparison
plt.tight_layout()
plt.show()

# Optionally, save the heatmap overlay
cv2.imwrite('route29_heatmap_player_overlay.png', overlay)
