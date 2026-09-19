from types import SimpleNamespace

import numpy as np
import pytest

from starVLA.dataloader.gr00t_lerobot import video


class FakeDecoder:
    def __init__(self, frames):
        self._frames = iter(frames)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._frames)

    def close(self):
        self.closed = True


class FakeFrame:
    def __init__(self, pts, value=0, error=None):
        self.pts = pts
        self.value = value
        self.error = error

    def to_ndarray(self, format):
        assert format == "rgb24"
        if self.error is not None:
            raise self.error
        return np.full((2, 3, 3), self.value, dtype=np.uint8)


class FakeContainer:
    def __init__(self, frames):
        stream = SimpleNamespace(time_base=0.1)
        self.streams = SimpleNamespace(video=[stream])
        self.frames = frames
        self.decoders = []
        self.closed = False

    def seek(self, *args, **kwargs):
        pass

    def decode(self, video):
        assert video == 0
        decoder = FakeDecoder(self.frames)
        self.decoders.append(decoder)
        return decoder

    def close(self):
        self.closed = True


@pytest.mark.parametrize("backend", ["pyav", "torchvision_av"])
def test_timestamp_decoder_closes_resources(monkeypatch, backend):
    container = FakeContainer([FakeFrame(0, value=1), FakeFrame(1, value=2), FakeFrame(2)])
    monkeypatch.setattr(video.av, "open", lambda _: container)

    frames = video.get_frames_by_timestamps("video.mp4", [0.1], video_backend=backend)

    assert frames.shape == (1, 2, 3, 3)
    assert frames[0, 0, 0, 0] == 2
    assert container.closed
    assert all(decoder.closed for decoder in container.decoders)


def test_timestamp_decoder_closes_resources_on_conversion_error(monkeypatch):
    container = FakeContainer([FakeFrame(0, error=MemoryError("decode failed"))])
    monkeypatch.setattr(video.av, "open", lambda _: container)

    with pytest.raises(MemoryError, match="decode failed"):
        video.get_frames_by_timestamps("video.mp4", [0.0], video_backend="torchvision_av")

    assert container.closed
    assert all(decoder.closed for decoder in container.decoders)
