###############################################################################
#  Decode uploaded audio clips (/humanaudio, put_audio_filepath)
###############################################################################

from typing import BinaryIO, Union

import av
import numpy as np
import soundfile as sf

AudioSource = Union[str, BinaryIO]


def decode_audio(source: AudioSource) -> tuple[np.ndarray, int]:
    """Decode a clip to (samples, sample_rate), shaped like sf.read: 1-D mono, else (n, channels).

    MP3 goes through PyAV (FFmpeg). libsndfile only estimates an MP3's length when it has no
    Xing/Info header (as when ffmpeg writes it to a pipe), and sf.read stops at that estimate,
    silently dropping the end of the clip. Everything else keeps using soundfile.
    """
    if sf.info(_rewound(source)).format != "MP3":
        return sf.read(_rewound(source))
    return _decode_with_pyav(_rewound(source))


def _rewound(source: AudioSource) -> AudioSource:
    if not isinstance(source, str):
        source.seek(0)
    return source


def _decode_with_pyav(source: AudioSource) -> tuple[np.ndarray, int]:
    with av.open(source) as container:
        stream = container.streams.audio[0]
        # planar float32, same layout and rate: to_ndarray() gives (channels, samples)
        resampler = av.AudioResampler(format="fltp")
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(out.to_ndarray() for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray() for out in resampler.resample(None))
        rate = stream.rate
    if not chunks:
        return np.zeros(0, dtype=np.float32), rate
    samples = np.concatenate(chunks, axis=1).T
    return (samples[:, 0] if samples.shape[1] == 1 else samples), rate
