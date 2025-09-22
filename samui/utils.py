# Copyright 2025 Maria Taktasheva, Helena Pankov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms


def target_size_from_min_dimension(
    image: np.ndarray | Image.Image, min_dimension: int
) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        width, height = image.size
    else:
        height, width = image.shape[:2]

    if height < width:
        new_height = min_dimension
        new_width = int(width * new_height / height)
    else:
        new_width = min_dimension
        new_height = int(height * new_width / width)

    return new_width, new_height


def put_text(frame, position, text, color=(0, 0, 255)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def load_video_frames(frames_dir: str | Path):
    frames_dir = Path(frames_dir)

    jpeg_file_paths = [
        path
        for path in frames_dir.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg"]
    ]
    jpeg_file_paths.sort(key=lambda path: path.stem)

    frames = [cv2.imread(str(path)) for path in jpeg_file_paths]
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    return frames


def open_image(image_path: str | Path) -> Image.Image:
    image = Image.open(image_path)
    icc = image.info.get("icc_profile")
    if icc:
        image = ImageCms.profileToProfile(image, icc, "sRGB")
    return image
