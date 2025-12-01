# Gaussian Splatting Navigation & Cinematic Renderer (ROS 2 + Docker)

This project runs a full ROS 2 Humble pipeline for:

- Loading a Gaussian Splatting scene (`.ply`)
- Converting it into a 2D occupancy map for Nav2
- Navigating a virtual **camera** through the environment
- Recording the camera trajectory from TF
- Rendering a **cinematic video** along the recorded path using **gsplat** (in progres)

Everything runs inside a Docker container.

IMPORTNT: add `ConferenceHall.ply` in `data` folder

### You can check `results/cinematic_output.mp4`

---

## 1. Prerequisites

On the **host**:

- Docker
- Docker Compose (v2)
- NVIDIA drivers with docker (for GPU rendering with gsplat)
- X11 running (for RViz2 GUI)

For X11 on Linux you will usually also need:

```bash
xhost +local:docker
````

---

## 2. Start the Docker Container

Build the image and start the `terminal` container. From the root of this repository:

```bash
docker compose up --build terminal
```

Open an interactive shell inside the running container. In a separate terminal:

```bash
docker compose exec terminal bash
```

> *note:* If you prefer the CUDA-enabled service (e.g., `terminal-cuda`), just replace
> `terminal` with `terminal-cuda` in the commands above.

---

## 3. Build the ROS 2 Workspace

Inside the container shell:

```bash
# Build the workspace
colcon build

# Source the workspace
source install/setup.bash
```

---

## 4. Launch the System

To start the full pipeline (Nav2, map generation, scene renderer, camera controller,
path recorder, gsplat node, RViz2):

```bash
ros2 launch mook_gaussian_bringup bringup.launch.py
```

> *note:* If you have problems with CUDA in container use `cudaless_bringup.launch.py` launch file, just replace

This will:

* Load your Gaussian Splatting `.ply` scene
* Generate a 2D occupancy map and publish it as `/scene_map`
* Start Nav2 (planner, controller, costmaps, etc.)
* Start a virtual camera (`camera` frame) controlled by `/cmd_vel`
* Start TF-based camera path recording
* Start the gsplat cinematic renderer node
* Open RViz2 (inside the container, but displayed on your host via X11)
* Also it will put row video recordings in `results` folder
---

## 5. Navigating the Camera in RViz

In RViz2:

1. Make sure the **Fixed Frame** is set to `map`.
2. Display:

   * `Map` (topic: `/scene_map`)
   * `TF`
   * `Pose` / `RobotModel` or something to visualize `camera`
3. Use the **2D Pose Estimate** tool to set the initial camera pose:

   * Frame: `map`
4. Use **Nav2 Goal** / **2D Nav Goal** to send a navigation goal:

   * Nav2 will plan a path in the `map` frame
   * `controller_server` will publish `/cmd_vel`
   * `camera_controller` will move the `camera` frame along this path

While you drive or send goals, the camera poses are continuously recorded from TF.

---

## 6. Recording and Saving the Camera Path

The **CameraPathRecorder** node runs automatically from the launch file and
samples TF (`map -> camera`) at a fixed rate.

To **save** the current path to disk:

```bash
ros2 service call /save_camera_path std_srvs/srv/Trigger "{}"
```

You should see a message like:

> Saved N poses to paths/camera_path.json

The file will be created inside the container at:

```text
/home/mobile/ros2_ws/paths/camera_path.json
```

To clear the in-memory path and start a fresh recording:

```bash
ros2 service call /reset_camera_path std_srvs/srv/Trigger "{}"
```

---

## 7. Rendering the Cinematic Video with gsplat

Once you’re happy with the path (you’ve driven around / used Nav2 and saved it):

1. Make sure the workspace is sourced:

   ```bash
   source install/setup.bash
   ```

2. Call the cinematic render service:

   ```bash
   ros2 service call /render_cinematic std_srvs/srv/Trigger "{}"
   ```

The **gsplat renderer node** will:

* Load `paths/camera_path.json`
* Load your Gaussian Splatting scene (e.g. `src/data/ConferenceHall.ply`)
* Render each frame along the recorded poses
* Write the final video to something like:

```text
/home/mobile/ros2_ws/cinematic_output.mp4
```

(Exact path is defined by the node’s parameters; by default it’s `cinematic_output.mp4` in the workspace.)

---

## 8. Copying the Video from the Container

From the **host**, you can copy the rendered video out:

```bash
# Find the container name (e.g., <project>_terminal_1)
docker ps

# Example:
docker cp <container_name>:/home/mobile/ros2_ws/cinematic_output.mp4 ./cinematic_output.mp4
```

Now you can play the video on your host with any media player.

---

## 10. Notes

* All paths in this README assume the user `mobile` and workspace at `/home/mobile/ros2_ws`.
* The repository is bind-mounted into the container at `/home/mobile/ros2_ws/src` via `docker-compose.yml`.
* If RViz2 cannot open a window, check:

  * `DISPLAY` is set on the host and passed to Docker
  * `xhost +local:docker` has been run
  * X11 socket is mounted (`/tmp/.X11-unix`)

