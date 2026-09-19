# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import av
import cv2
import numpy as np

import torch  # noqa: F401 # isort: skip
import torchvision  # noqa: F401 # isort: skip

# Import decord with graceful fallback
try:
    import decord  # noqa: F401

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import torchcodec

    TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHCODEC_AVAILABLE = False


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_at(indices=indices).data.numpy()
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "pyav":
        # Use PyAV directly without torchvision wrapper
        container = None
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            fps = float(stream.average_rate) if stream.average_rate else float(stream.guessed_rate)
            
            loaded_frames = []
            
            for target_idx in indices:
                # Convert frame index to timestamp, then to pts
                target_ts = target_idx / fps
                target_pts = int(target_ts / time_base)
                
                # Seek to the target position
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                
                closest_frame = None
                closest_idx_diff = float('inf')
                
                for frame in container.decode(video=0):
                    current_ts = float(frame.pts * time_base)
                    current_idx = int(round(current_ts * fps))
                    current_diff = abs(current_idx - target_idx)
                    
                    if current_diff < closest_idx_diff:
                        closest_idx_diff = current_diff
                        closest_frame = frame
                    
                    # If we've passed the target and diff is increasing, stop
                    if current_idx > target_idx and current_diff > closest_idx_diff:
                        break
                    
                    # Stop if we're significantly past the target
                    if current_idx > target_idx + 5:
                        break
                
                if closest_frame is not None:
                    frame_data = closest_frame.to_ndarray(format="rgb24")
                    loaded_frames.append(frame_data)
                else:
                    raise ValueError(f"Unable to find frame at index {target_idx}")
            
            frames = np.array(loaded_frames)
            return frames
            
        finally:
            if container is not None:
                container.close()
    else:
        raise NotImplementedError


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """Get frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        # For some GPUs, AV format data cannot be read
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        return decoder.get_frames_played_at(seconds=timestamps).data.numpy()
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend in {"pyav", "torchvision_av"}:
        # torchvision.io.VideoReader does not expose a reliable close API for
        # its PyAV container. Use PyAV directly so every decoder is closed
        # deterministically between samples.
        container = None
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            loaded_frames = []

            for target_ts in timestamps:
                target_ts = float(target_ts)
                target_pts = int(target_ts / time_base)
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)

                closest_frame = None
                closest_ts_diff = float('inf')

                decoder = container.decode(video=0)
                try:
                    for frame in decoder:
                        if frame.pts is None:
                            continue
                        current_ts = float(frame.pts * time_base)
                        current_diff = abs(current_ts - target_ts)

                        if current_diff < closest_ts_diff:
                            closest_ts_diff = current_diff
                            closest_frame = frame

                        if current_ts > target_ts and current_diff > closest_ts_diff:
                            break
                        if current_ts > target_ts + 1.0:
                            break
                finally:
                    close_decoder = getattr(decoder, "close", None)
                    if close_decoder is not None:
                        close_decoder()

                if closest_frame is not None:
                    frame_data = closest_frame.to_ndarray(format="rgb24")
                    loaded_frames.append(frame_data)
                else:
                    raise ValueError(f"Unable to find frame at timestamp {target_ts}")

            return np.stack(loaded_frames)
        finally:
            if container is not None:
                container.close()
    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Get all frames from a video.
    Args:
        video_path (str): Path to the video file.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
        video_backend_kwargs (dict, optional): Keyword arguments for the video backend.
        resize_size (tuple[int, int], optional): Resize size for the frames. Defaults to None.
    """
    if video_backend == "decord":
        if not DECORD_AVAILABLE:
            raise ImportError("decord is not available.")
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    elif video_backend == "torchcodec":
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec is not available.")
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        frames = decoder.get_frames_at(indices=range(len(decoder)))
        return frames.data.numpy(), frames.pts_seconds.numpy()
    elif video_backend == "pyav":
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)
        frames = np.array(frames)
    elif video_backend == "torchvision_av":
        # set backend and reader
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        frames = []
        for frame in reader:
            frames.append(frame["data"].numpy())
        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames
