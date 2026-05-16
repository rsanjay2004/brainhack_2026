import os, cv2, numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

TOPIC = (
    "/world/roboverse/model/x500_vision_0"
    "/link/camera_link/sensor/IMX214/image"
)
os.makedirs("dataset/raw", exist_ok=True)
count = 0

def cb(msg):
    global count
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        (msg.height, msg.width, 3)
    )
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    path = f"dataset/raw/frame_{count:05d}.jpg"
    cv2.imwrite(path, bgr)
    count += 1
    if count % 20 == 0:
        print(f"Saved {count} frames")

node = Node()
node.subscribe(Image, TOPIC, cb)
input("Fly the drone around barrels — press Enter to stop\n")
print(f"Total: {count} frames saved to dataset/raw/")