"""Stream the recorder's embedded JPEGs to CFR MP4s (no ROS, GPU or frame resampling)."""

import io
import json
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def jpeg_rows(parquet: pq.ParquetFile, column: str) -> Iterator[bytes]:
    """One iterator per camera, continued (not restarted) across pause boundaries."""
    for batch in parquet.iter_batches(columns=[column], batch_size=16, use_threads=False):
        images = batch.column(0)
        if not pa.types.is_struct(images.type) or images.type.get_field_index('bytes') < 0:
            raise ValueError(f'{column}: expected HF Image struct with embedded bytes')
        payloads = images.field('bytes')
        if images.null_count or payloads.null_count or not pa.types.is_binary(payloads.type):
            raise ValueError(f'{column}: missing or invalid embedded JPEG payload')
        for item in payloads:
            yield item.as_py()


def check_video(path: Path, fps: int, size: tuple[int, int], count: int) -> None:
    """Check decoded presentation order and integer PTS, not just the container header."""
    cmd = ['ffprobe', '-v', 'error', '-threads', '2', '-select_streams', 'v:0', '-show_frames', '-show_streams',
           '-show_entries', 'stream=codec_name,pix_fmt,width,height,time_base,avg_frame_rate:'
           'frame=best_effort_timestamp', '-of', 'json', str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode or result.stderr.strip():
        raise ValueError(f'{path}: ffprobe failed: {result.stderr.strip()}')
    data = json.loads(result.stdout)
    stream, = data['streams']
    if (stream['codec_name'], stream['pix_fmt'], stream['width'], stream['height']) != (
            'h264', 'yuv420p', *size) or Fraction(stream['avg_frame_rate']) != fps:
        raise ValueError(f'{path}: unexpected video stream: {stream}')
    frames = data.get('frames', [])
    if len(frames) != count:
        raise ValueError(f'{path}: decoded {len(frames)} frames; expected {count}')
    tb = Fraction(stream['time_base'])
    for i, frame in enumerate(frames):
        if int(frame['best_effort_timestamp']) * tb.numerator * fps != i * tb.denominator:
            raise ValueError(f'{path}: frame {i} is not at {i}/{fps} seconds')


def write_video(path: Path, frames: Iterator[bytes], fps: int, size: tuple[int, int], n: int) -> None:
    """Consume exactly n JPEG rows. Identical payloads still become distinct timed frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-xerror', '-filter_threads', '2',
           '-f', 'image2pipe', '-framerate', str(fps), '-threads', '2', '-c:v', 'mjpeg', '-i', 'pipe:0',
           '-map', '0:v:0', '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'veryfast',
           '-crf', '18', '-pix_fmt', 'yuv420p', '-g', str(fps), '-bf', '0', '-vsync', '0',
           '-video_track_timescale', str(fps), '-movflags', '+faststart', str(path)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        try:
            with proc.stdin as pipe:
                for i in range(n):
                    blob = next(frames, None)
                    if not blob or not blob.startswith(b'\xff\xd8') or not blob.endswith(b'\xff\xd9'):
                        raise ValueError(f'{path}, row {i}: missing/incomplete JPEG')
                    # Check each header: FFmpeg could otherwise silently resize a changed view.
                    with Image.open(io.BytesIO(blob)) as image:
                        if (image.format, image.size, image.mode) != ('JPEG', size, 'RGB'):
                            raise ValueError(f'{path}, row {i}: JPEG {image.size}/{image.mode}, expected {size}/RGB')
                    pipe.write(blob)
        except BrokenPipeError as exc:
            raise RuntimeError(f'{path}: FFmpeg stopped accepting frames (see stderr above)') from exc
        except BaseException:
            proc.kill()
            raise
        if proc.wait():
            raise RuntimeError(f'{path}: FFmpeg exited with status {proc.returncode}')
    check_video(path, fps, size, n)
