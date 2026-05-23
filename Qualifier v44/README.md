 Qualifier v51 Right Wall — Instructions

This guide explains the dependencies, how to run the code, and how the `Qualifier v51 Right wall` solution works.

## 1. What this version does

`qualifier_right_wall_v51.py` is the main competition script for the **right-wall qualifier**.

The script controls a PX4/Gazebo drone that must autonomously explore the arena while detecting and counting red and yellow barrels. It uses a right-wall-following navigation strategy as the default behaviour, while keeping the existing YOLO barrel detector, RGB/depth subscribers, occupancy-grid mapping, tracking, and safety logic.

In simple terms, the drone:

1. Takes off to a fixed altitude.
2. Subscribes to RGB and depth camera topics from the simulator.
3. Uses the depth image to estimate free space, walls, corners, and obstacles.
4. Follows the right wall to move through the environment consistently.
5. Uses YOLO to detect red and yellow barrels from RGB frames.
6. Projects detections into approximate world positions using depth.
7. Merges repeated detections so the same barrel is not counted multiple times.
8. Saves output logs, annotated images, and map/debug files into an output folder.

## 2. Folder to use

After unzipping the project, use this folder:

```bash
cd "brainhack_2026-FINAL/Qualifier v51 Right wall"
```

Expected structure:

```text
Qualifier v51 Right wall/
├── qualifier_right_wall_v51.py
├── drone_control.py
├── depth_receiver.py
├── get_position_with_task.py
├── AvoidancePlanner.py
└── models/
    └── barrel_best.pt
```

Keep the helper files in the same folder as `qualifier_right_wall_v51.py`, because the main script imports them directly.

## 3. Dependencies

### Python version

Recommended:

```text
Python 3.10 or Python 3.11
```

### Python packages

Install the core Python packages:

```bash
pip3 install numpy opencv-python mavsdk ultralytics
```

These packages are used for:

| Package | Why it is needed |
|---|---|
| `numpy` | Matrix operations, depth image processing, coordinate calculations, and occupancy grid updates. |
| `opencv-python` | RGB image handling, annotation drawing, colour checking, and image saving. |
| `mavsdk` | Communicating with the PX4 drone and sending takeoff/offboard velocity commands. |
| `ultralytics` | Loading and running the YOLO model for barrel detection. |
| 'gz.transport' | Primarily designed for robotics applications and is the core messaging system for the Gazebo robot simulator. | sudo apt install libgz-transport13-dev
| 'gz.mesgs10' | Standard Data Validation | sudo apt install libgz-msgs

### Gazebo / GZ Python bindings

The code also imports:

```python
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
```

These come from the Gazebo/GZ simulator Python bindings, not from normal `pip` packages. They must already be available in the provided BrainHack VM/environment.

The script expects the simulator to publish these default camera topics:

```text
RGB topic:   /world/roboverse/model/x500_depth_0/link/camera_link/sensor/IMX214/image
Depth topic: /depth_camera
```

### Model file

The trained model should be here:

```text
models/barrel_best.pt
```

The default run command already points to this file.

## 4. How to run the code

### Step 1: Start the simulator

Start the BrainHack/PX4/Gazebo simulation first. The drone must be ready and MAVSDK must be able to connect to PX4.

The code expects MAVSDK/PX4 to be reachable through:

```text
udpin://0.0.0.0:14540
```

This connection is handled inside `drone_control.py`.

### Step 2: Go to the v51 folder

```bash
cd "brainhack_2026-FINAL/Qualifier v51 Right wall"
```

### Step 3: Confirm the model exists

```bash
ls models/barrel_best.pt
```

If the file exists, continue. If not, place `barrel_best.pt` inside the `models` folder or pass the correct model path using `--model`.

### Step 4: Run a short test first

Use a short run to check that the simulator, drone connection, camera topics, and YOLO model are working:

```bash
python3 qualifier_right_wall_v51.py \
  --model models/barrel_best.pt \
  --duration-s 60 \
  --no-land \
  --takeoff-altitude-m 3.2 \
  --max-flying-height-m 7.0 \
  --conf 0.35
```

During the test, check that:

- The drone connects successfully.
- RGB frames are received.
- Depth frames are received.
- YOLO loads `models/barrel_best.pt`.
- The drone starts right-wall navigation.
- An output folder is created.

### Step 5: Run the full qualifier command

Recommended competition command:

```bash
python3 qualifier_right_wall_v51.py \
  --model models/barrel_best.pt \
  --duration-s 300 \
  --no-land \
  --takeoff-altitude-m 3.2 \
  --max-flying-height-m 7.0 \
  --conf 0.35 \
  --merge-radius-m 0.70 \
  --duplicate-suppression-radius-m 1.10 \
  --min-track-hits 2
```

## 5. Important run arguments

| Argument | Recommended value | Meaning |
|---|---:|---|
| `--model` | `models/barrel_best.pt` | Path to the trained YOLO barrel detector. |
| `--duration-s` | `300` | Runs for 5 minutes. |
| `--no-land` | enabled | Prevents automatic landing at the end. Remove this if auto-land is required. |
| `--takeoff-altitude-m` | `3.2` | Takeoff altitude above the starting position. |
| `--max-flying-height-m` | `7.0` | Highest altitude allowed for vertical obstacle avoidance. |
| `--conf` | `0.35` | YOLO confidence threshold. Lower detects more; higher reduces false positives. |
| `--merge-radius-m` | `0.70` | Merges nearby detections into the same barrel track. |
| `--duplicate-suppression-radius-m` | `1.10` | Prevents repeated counting around confirmed barrel tracks. |
| `--min-track-hits` | `2` | Requires at least two detections before a barrel is counted. |
| `--navigation-mode` | `right_wall` | Uses the v51 right-wall navigation strategy. |

## 6. Output files

By default, output is saved to:

```text
qualifier_right_wall_v51_output/
```

Typical outputs include:

```text
occupancy_grid.png
annotated detection images
saved RGB context images
barrel tracking/count logs
mission/debug files
```

To use a different output folder:

```bash
python3 qualifier_right_wall_v51.py --output-dir my_output_folder
```

## 7. How the solution solves the problem

The qualifier problem requires the drone to move around the environment safely and count the barrels accurately. This script solves it by combining four main systems.

### 7.1 Right-wall navigation

The default navigation mode is:

```bash
--navigation-mode right_wall
```

Instead of randomly exploring, the drone follows the right wall. This gives the drone a consistent rule for moving through corridors and around corners. It uses depth readings from the front, left, and right sectors to decide whether to move forward, correct its distance from the right wall, turn right into an opening, or turn left when the front is blocked.

This is useful because wall-following reduces decision loops and helps the drone cover maze-like areas in a predictable way.

### 7.2 Depth-based obstacle avoidance

The depth camera is used to estimate how far obstacles are from the drone. The code checks whether the front is blocked, whether the right side is too close, and whether there is enough space to continue.

If the drone gets too close to an obstacle, it slows down, turns, or uses recovery logic. The script also supports vertical avoidance using `--enable-vertical-avoidance`, allowing the drone to climb or descend within safe altitude limits when needed.

### 7.3 YOLO barrel detection

The RGB camera frames are passed into a YOLO model loaded from:

```text
models/barrel_best.pt
```

The model detects red and yellow barrels. The script supports both class-name matching and class-ID matching:

```text
red class IDs:    0
yellow class IDs: 1
```

It also includes a colour-override safety check. This means that if the YOLO class label is uncertain but the crop clearly looks red or yellow, the script can still classify the barrel correctly.

### 7.4 Barrel tracking and duplicate suppression

A barrel may appear in multiple frames while the drone moves past it. To avoid counting the same barrel repeatedly, the script estimates the barrel position using the detection box and depth data, then merges nearby detections into the same track.

The main duplicate-control settings are:

```bash
--merge-radius-m 0.70
--duplicate-suppression-radius-m 1.10
--min-track-hits 2
```

This means a barrel normally needs to be seen at least twice before it contributes to the final count, and detections close to an already confirmed barrel are suppressed.

## 8. Optional tuning

### If the drone is too close to the right wall

```bash
--right-wall-target-m 1.50
```

### If the drone is too far from the right wall

```bash
--right-wall-target-m 1.10
```

### If the drone turns right too easily

```bash
--right-wall-open-m 2.80 \
--right-wall-open-confirm-frames 3
```

### If the drone misses right turns

```bash
--right-wall-open-m 2.10
```

### If the drone moves too fast

```bash
--right-wall-speed-m-s 0.70
```

### If the drone misses small or far barrels

```bash
--yolo-imgsz 640 \
--conf 0.25
```

### If there are too many false barrel detections

```bash
--conf 0.45 \
--min-track-hits 3
```

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'mavsdk'`

```bash
pip3 install mavsdk
```

### `ModuleNotFoundError: No module named 'ultralytics'`

```bash
pip3 install ultralytics
```

### `ModuleNotFoundError: No module named 'cv2'`

```bash
pip3 install opencv-python
```

### `ModuleNotFoundError: No module named 'gz'`

This usually means the code is not being run inside the correct Gazebo/BrainHack environment. Use the provided VM/container/environment where the GZ Python bindings are installed.

### `No such file or directory: models/barrel_best.pt`

Check that the model file exists:

```bash
ls "models/barrel_best.pt"
```

If the model is elsewhere:

```bash
python3 qualifier_right_wall_v51.py --model /path/to/barrel_best.pt
```

### Drone does not connect

Make sure PX4/Gazebo is already running and that MAVSDK can connect through:

```text
udpin://0.0.0.0:14540
```

### RGB or depth frames are not received

Check that the simulator is publishing the expected topics. If your topic names are different, override them:

```bash
python3 qualifier_right_wall_v51.py \
  --rgb-topic YOUR_RGB_TOPIC \
  --depth-topic YOUR_DEPTH_TOPIC
```

### Drone keeps stopping near obstacles

Try reducing speed or increasing clearance:

```bash
--right-wall-speed-m-s 0.70 \
--right-wall-target-m 1.50 \
--emergency-stop-m 1.30
```

## 10. Final checklist

Before the actual qualifier run, confirm:

- [ ] Simulator/PX4 is running.
- [ ] You are inside `brainhack_2026-FINAL/Qualifier v51 Right wall`.
- [ ] `models/barrel_best.pt` exists.
- [ ] Python dependencies are installed.
- [ ] The GZ/Gazebo Python bindings are available.
- [ ] RGB and depth topics are correct.
- [ ] A 60-second test run works.
- [ ] The 300-second command is used for the full qualifier run.
