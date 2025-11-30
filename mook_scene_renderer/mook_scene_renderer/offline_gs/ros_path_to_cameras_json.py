#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np


def quat_xyzw_to_R(q):
    x, y, z, w = q
    # Нормируем
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=float)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )
    return R


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_path", required=True,
                        help="camera_path.json, записанный ROS-нодой")
    parser.add_argument("--cameras_json", required=True,
                        help="выходной cameras.json для JSONCameraDataset")
    args = parser.parse_args()

    camera_path = Path(args.camera_path)
    cameras_json = Path(args.cameras_json)

    with camera_path.open("r") as f:
        data = json.load(f)

    W = int(data["image_width"])
    H = int(data["image_height"])
    hfov_deg = float(data["horizontal_fov_deg"])
    frames = data["frames"]

    hfov = math.radians(hfov_deg)
    fx = W / (2.0 * math.tan(hfov / 2.0))
    fy = fx * (H / W)
    vfov = 2.0 * math.atan(H / (2.0 * fy))

    cameras = []

    for cam_id, frame in enumerate(frames):
        pos = frame["position"]
        quat = frame["orientation_xyzw"]
        R = quat_xyzw_to_R(quat)  # 3x3
        T = np.array(pos, dtype=float)

        # ВАЖНО:
        # Здесь нужно привести формат к тому, что ожидает gaussian_splatting.JSONCameraDataset.
        # Проще всего:
        #  1) запустить любой пример с ColmapCameraDataset
        #  2) вызвать dataset.save_cameras("sample_cameras.json")
        #  3) посмотреть структуру и переиспользовать ключи.
        #
        # Ниже — типичный пример структуры CameraInfo,
        # НО ТЫ ДОЛЖЕН ПРОВЕРИТЬ ЕЁ ПО СВОЕМУ sample_cameras.json:

        cam = {
            "id": cam_id,
            "R": R.tolist(),          # rotation 3x3
            "T": T.tolist(),          # translation 3
            "FovX": float(hfov),
            "FovY": float(vfov),
            "width": W,
            "height": H,
            # Если нужно — добавь "image_name", "image_path", "uid", "sharpness" и прочие поля,
            # ориентируясь на реальный cameras.json.
        }

        cameras.append(cam)

    with cameras_json.open("w") as f:
        json.dump(cameras, f, indent=2)

    print(f"Saved {len(cameras)} cameras to {cameras_json}")


if __name__ == "__main__":
    main()
