"""Display/encode RGB comparisons locally. No GPU, web server, network requests or video re-timing."""
import subprocess
from functools import lru_cache
from itertools import chain
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ws.src.evaluation.pillow_open_loop import FPS

TOP, FOOT = 36, 34


@lru_cache(maxsize=1)
def font():
    try:
        return ImageFont.truetype('DejaVuSans.ttf', 18)
    except OSError:
        return ImageFont.load_default()


def side_by_side(gt, pred, step, source, anchor, offset):
    if (gt.shape != pred.shape or gt.ndim != 3 or gt.shape[-1] != 3
            or gt.dtype != np.uint8 or pred.dtype != np.uint8):
        raise ValueError(f'实录/预测 RGB 几何或类型不一致：{gt.shape} {gt.dtype} / {pred.shape} {pred.dtype}；不强制拉伸。')
    h, w = gt.shape[:2]
    canvas = Image.new('RGB', (2*w, TOP+h+FOOT), '#162430')
    canvas.paste(Image.fromarray(gt), (0, TOP))
    canvas.paste(Image.fromarray(pred), (w, TOP))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 6), 'RECORDED  |  same source frame', font=font(), fill='#a3ecdb')
    title = 'ANCHOR RECONSTRUCTION  |  not predicted' if offset == 0 else f'DREAMZERO FORECAST  |  +{offset*1000//FPS} ms'
    draw.text((w+10, 6), title, font=font(), fill='#ffe29c')
    frame = anchor + offset
    draw.text((10, TOP+h+4), f'checkpoint {step}   |   source {source}   |   window anchor {anchor}   |   '
              f'source frame {frame} ({frame/FPS:.2f} s)   |   offset +{offset}', font=font(), fill='white')
    return np.asarray(canvas)


def check_encoder():
    proc = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True)
    if proc.returncode or not any(row.split()[1:2] == ['libx264'] for row in proc.stdout.splitlines()):
        raise ValueError('需要带 libx264 编码器的 ffmpeg（与数据转换时相同）')


def encode_mp4(path, frames, rate):
    frames = iter(frames)
    first = next(frames, None)
    if first is None:
        raise ValueError('没有对照帧')
    h, w, c = first.shape
    if c != 3 or w % 2 or h % 2:
        raise ValueError(f'Expected even dimensions for H.264/yuv420p: {first.shape}')
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-filter_threads', '2',
           '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{w}x{h}', '-framerate', str(rate),
           '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-preset', 'veryfast',
           '-pix_fmt', 'yuv420p', '-bf', '0', '-g', '9', '-video_track_timescale', str(rate.numerator),
           '-movflags', '+faststart', str(path)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        with proc.stdin as pipe:
            for n, frame in enumerate(chain((first,), frames), start=1):
                if frame.shape != first.shape or frame.dtype != np.uint8:
                    raise ValueError('编码序列的尺寸或像素类型不一致')
                pipe.write(np.ascontiguousarray(frame).data)
    if proc.returncode:
        raise ValueError(f'ffmpeg 编码失败（退出码 {proc.returncode}）')
    return n


def page(meta):
    """Only numeric window IDs go into markup; use a sibling MP4 so file:// works offline."""
    from fractions import Fraction
    windows = meta['windows']
    options = ''.join(f'<option value="{w["first_output_frame"]}">原始帧 {w["source_anchor"]}–{w["source_frames"][-1]}</option>'
                      for w in windows)
    return '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>翻枕头 · 预测视频对照</title>
<style>body{font:17px/1.6 system-ui,sans-serif;max-width:1280px;margin:24px auto;padding:0 14px;color:#182b35;background:#f6f8f9}h1{font-size:1.45rem}video{width:100%;background:#162430}button,select{font:inherit;margin:4px;padding:3px 8px}p{margin:.65em 0}</style>
<h1>翻枕头 · __CHECKPOINT__ · 留出录制 __SOURCE__</h1>
<p><b>左：真实录制　｜　右：DreamZero</b>。两边都是左上顶视、右上右腕、左下左腕、右下未用。先看夹爪是否带动枕角，再看红胶带在哪一帧出现。</p>
<video id="clip" controls loop playsinline preload="auto" src="comparison.mp4">请用本地播放器打开同目录的 comparison.mp4。</video>
<div><label>跳到窗口 <select id="window">__OPTIONS__</select></label>
<button type="button" id="prev">上一帧</button><button type="button" id="next">下一帧</button>
<label>速度 <select id="speed"><option value="0.25">0.25×</option><option value="0.5" selected>0.5×</option><option value="1">1×（名义录制速度）</option></select></label></div>
<p>每窗从真实画面、当前关节状态和同一指令重新起步。右侧第 1 张是起点的 VAE 重建，不算预测；后 8 张是未来 60–480 ms 的预测。相邻窗口的边界可出现两次：先是上一窗的预测，后是下一窗的真实条件重建。不是整段自主翻面，也不是真机成功率。</p>
<script>
const v=document.getElementById('clip'), pick=document.getElementById('window'), speed=document.getElementById('speed');
const fps=__RATE__, count=__COUNT__, per=__PER__;
function position(){return Math.min(count-1,Math.floor(v.currentTime*fps+0.00001));}
function seek(i){v.pause();v.currentTime=(Math.max(0,Math.min(count-1,i))+0.1)/fps;}
pick.onchange=()=>seek(Number(pick.value));
v.ontimeupdate=()=>{pick.selectedIndex=Math.floor(position()/per);};
v.onloadedmetadata=speed.onchange=()=>{v.playbackRate=Number(speed.value);};v.playbackRate=Number(speed.value);
document.getElementById('prev').onclick=()=>seek(position()-1);document.getElementById('next').onclick=()=>seek(position()+1);
</script></html>'''.replace('__CHECKPOINT__', Path(meta['checkpoint']).name).replace('__SOURCE__', str(meta['source_id'])).replace(
        '__OPTIONS__', options).replace('__RATE__', str(float(Fraction(meta['output_fps'])))).replace(
        '__COUNT__', str(len(windows)*meta['frames_per_window'])).replace('__PER__', str(meta['frames_per_window']))
