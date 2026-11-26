import io
import struct
import unittest

import numpy as np
import soundfile as sf

from utils.audio_decode import decode_audio

SR = 22050

# MPEG layer III bitrates in kbps: MPEG-1, then MPEG-2/2.5
_BITRATES = {
    3: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
    2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
}
_RATES = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def speechlike(seconds: float, rate: int = SR) -> np.ndarray:
    """Tone bursts with pauses, so a VBR encoder uses different frame sizes."""
    rng = np.random.default_rng(1)
    t = np.arange(int(seconds * rate)) / rate
    envelope = (np.sin(2 * np.pi * 1.3 * t) > -0.2).astype(np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)
    return (tone * envelope).astype(np.float32)


def encode_mp3(samples: np.ndarray, rate: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="MP3")
    return buf.getvalue()


def without_length_header(mp3: bytes) -> bytes:
    """Drop the Xing/Info frame, as in an MP3 that ffmpeg wrote to a pipe."""
    pos = 0
    if mp3[:3] == b"ID3":
        pos = 10 + ((mp3[6] << 21) | (mp3[7] << 14) | (mp3[8] << 7) | mp3[9])
    header = struct.unpack(">I", mp3[pos:pos + 4])[0]
    assert header >> 21 == 0x7FF, "no MP3 frame sync"
    version = (header >> 19) & 3
    bitrate = _BITRATES[3 if version == 3 else 2][(header >> 12) & 15] * 1000
    rate = _RATES[version][(header >> 10) & 3]
    length = (144 if version == 3 else 72) * bitrate // rate + ((header >> 9) & 1)
    assert b"Xing" in mp3[pos:pos + length] or b"Info" in mp3[pos:pos + length]
    return mp3[:pos] + mp3[pos + length:]


class DecodeAudioTest(unittest.TestCase):
    def test_regression_headerless_mp3_decodes_to_full_length(self):
        # libsndfile estimates such a file's length from its first frame and sf.read stops there
        samples = speechlike(9.06)
        mp3 = without_length_header(encode_mp3(samples))
        decoded, rate = decode_audio(io.BytesIO(mp3))
        self.assertEqual(rate, SR)
        self.assertAlmostEqual(len(decoded) / rate, 9.06, delta=0.1)

    def test_mp3_with_header_decodes_to_full_length(self):
        decoded, rate = decode_audio(io.BytesIO(encode_mp3(speechlike(6.01))))
        self.assertAlmostEqual(len(decoded) / rate, 6.01, delta=0.1)

    def test_mp3_is_float32_mono_in_range(self):
        decoded, _ = decode_audio(io.BytesIO(encode_mp3(speechlike(1.0))))
        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(decoded.ndim, 1)
        self.assertLessEqual(float(np.max(np.abs(decoded))), 1.0)

    def test_stereo_mp3_keeps_channels_like_soundfile(self):
        mono = speechlike(1.0, 44100)
        decoded, _ = decode_audio(io.BytesIO(encode_mp3(np.stack([mono, -mono], axis=1), 44100)))
        self.assertEqual(decoded.ndim, 2)
        self.assertEqual(decoded.shape[1], 2)

    def test_wav_is_read_exactly(self):
        samples = speechlike(2.5, 16000)
        buf = io.BytesIO()
        sf.write(buf, samples, 16000, format="WAV", subtype="PCM_16")
        decoded, rate = decode_audio(io.BytesIO(buf.getvalue()))
        self.assertEqual((rate, len(decoded)), (16000, len(samples)))

    def test_file_path_is_accepted(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.mp3")
            with open(path, "wb") as f:
                f.write(without_length_header(encode_mp3(speechlike(3.0))))
            decoded, rate = decode_audio(path)
        self.assertAlmostEqual(len(decoded) / rate, 3.0, delta=0.1)

    def test_unreadable_data_raises(self):
        with self.assertRaises(Exception):
            decode_audio(io.BytesIO(b"not audio at all"))


if __name__ == "__main__":
    unittest.main()
