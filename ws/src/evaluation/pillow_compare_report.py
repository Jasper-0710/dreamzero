"""Encode one frame-locked mosaic: a single recorded reference and one forecast per checkpoint."""
from contextlib import ExitStack
from math import ceil, isqrt

import numpy as np
from PIL import Image, ImageDraw

from .export_pillow_video import OFFSETS, RATE
from .pillow_open_loop import FPS
from .pillow_video_report import font


def mosaic_frames(stage, steps, source, anchors):
    cols = isqrt(len(steps)) + 1  # ceil(sqrt(1 + N))
    for anchor in anchors:
        with ExitStack() as stack:
            files = [stack.enter_context((stage/f'checkpoint-{s}'/f'window-{anchor}.npyseq').open('rb')) for s in steps]
            for offset in OFFSETS:
                tiles = []
                for step, f in zip(steps, files):
                    gt, pred = np.load(f, allow_pickle=False)
                    if not tiles:
                        if gt.ndim != 3 or gt.shape[-1] != 3 or gt.dtype != np.uint8:
                            raise ValueError('实录不是 RGB uint8')
                        tiles.append(('RECORDED', gt))
                    elif not np.array_equal(gt, tiles[0][1]):
                        raise ValueError(f'checkpoint-{step} 实录预处理不一致，不能使用共同真值拼图')
                    if pred.shape != gt.shape or pred.dtype != np.uint8:
                        raise ValueError(f'checkpoint-{step} 预测 RGB 几何或像素类型不同')
                    tiles.append((f'ckpt {step} | ' + ('anchor reconstruction' if not offset else f'forecast +{offset*1000//FPS} ms'), pred))
                h, w = gt.shape[:2]
                canvas = Image.new('RGB', (cols*w, ceil(len(tiles)/cols)*(h+32)+32), '#17293b')
                draw = ImageDraw.Draw(canvas)
                for i, (label, rgb) in enumerate(tiles):
                    x, y = (i%cols)*w, (i//cols)*(h+32)
                    canvas.paste(Image.fromarray(rgb), (x, y+32))
                    draw.text((x+8, y+5), label, font=font(), fill='#a3ecdb' if i==0 else '#ffe29c')
                draw.text((8, canvas.height-27), f'source {source}   |   anchor {anchor}   |   source frame {anchor+offset}', font=font(), fill='white')
                yield np.asarray(canvas)
            if any(f.read(1) for f in files):
                raise ValueError(f'窗口 {anchor} 的帧数不一致')


def video_section(source, anchors):
    options = ''.join(f'<option value="{i*len(OFFSETS)}">{a} → {a+OFFSETS[-1]}</option>' for i,a in enumerate(anchors))
    return f'''<section class="card"><h2>同一时刻，各节点预测了什么？</h2>
<p>留出录制 {source}。左上是真实画面，其余按标签显示各节点。每窗第一张是起点重建，后 8 张是未来 60–480 ms 的预测。播放时所有格子严格同步。观察枕角是否被抬起，以及红胶带转向上的时刻。</p>
<video id="mosaic" controls playsinline preload="metadata" style="width:100%" src="comparison.mp4"></video>
<label>跳到原始帧 <select id="jump">{options}</select></label>
<button id="back">上一帧</button> <button id="forward">下一帧</button>
<label>速度 <select id="rate"><option>0.25</option><option selected>0.5</option><option>1</option></select> ×</label>
<script>(()=>{{const v=document.getElementById('mosaic'), pick=document.getElementById('jump'), speed=document.getElementById('rate');
const fps={float(RATE)}, n={len(anchors)*len(OFFSETS)}, pos=()=>Math.min(n-1, Math.floor(v.currentTime*fps+0.00001));
function seek(i){{v.pause(); v.currentTime=(0.1+Math.min(n-1,Math.max(0,i)))/fps;}}
pick.onchange=()=>seek(+pick.value); v.ontimeupdate=()=>pick.selectedIndex=Math.floor(pos()/{len(OFFSETS)});
v.onloadedmetadata=speed.onchange=()=>v.playbackRate=+speed.value;
document.getElementById('back').onclick=()=>seek(pos()-1); document.getElementById('forward').onclick=()=>seek(pos()+1);}})();</script></section>'''
