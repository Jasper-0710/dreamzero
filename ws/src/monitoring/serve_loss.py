#!/usr/bin/env python3
"""Read DreamZero's loss_log.jsonl in a browser. Loopback only; no GPU or third-party packages."""

import argparse
import json
import math
from html import escape
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, make_server

ROOT = Path(__file__).resolve().parents[3]  # repository containing checkpoints/
METRICS = {
    'loss': ('总损失', '#175eab'),
    'action_loss_avg': ('动作损失 · 滑动平均', '#007b6c'),
    'dynamics_loss_avg': ('视频预测损失 · 滑动平均', '#72509e'),
    'learning_rate': ('学习率', '#995118'),
}
STYLE = '''
*{box-sizing:border-box} body{margin:0;background:#f4f7fa;color:#182936;font:16px/1.5 system-ui,sans-serif}
main{max-width:1060px;margin:auto;padding:1.5rem} h1{font-size:1.65rem;margin:0 0 .6rem} h2{font-size:1.05rem;margin:0}
nav{display:flex;gap:.6rem;margin:1rem 0} a{color:#134e76;padding:.35rem .8rem;border:1px solid #a8bac7;border-radius:.5rem}
a[aria-current]{color:white;background:#174963} small,code,footer{color:#4c616c} code{overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,335px),1fr));gap:.85rem}
article{background:white;border:1px solid #cedae2;border-radius:.7rem;padding:1rem} .value{font-size:1.35rem;font-variant-numeric:tabular-nums}
svg{display:block;width:100%;height:auto} svg text{font:12px system-ui;fill:#405665} .warn{color:#91291c;font-weight:600}
footer{margin-top:1rem;font-size:.9rem}
'''


def snapshot(log):
    """The trainer appends separate records for different metrics at the same step."""
    series = {k: {} for k in METRICS}
    malformed, warnings = 0, []
    try:
        with log.open('rb') as stream:
            for raw in stream:
                if not raw.endswith(b'\n'):  # concurrent append; retry on next page refresh
                    break
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeError):
                    malformed += 1
                    continue
                if not isinstance(row, dict) or type(row.get('step')) is not int or row['step'] < 0:
                    malformed += 1
                    continue
                step = row['step']
                for key in METRICS.keys() & row.keys():
                    val = row[key]
                    if type(val) in (float, int) and math.isfinite(val):
                        series[key][step] = val
                    else:
                        series[key][step] = None  # gap, not a fabricated zero
                        warnings.append(f'{key} @ step {step}')
    except FileNotFoundError:
        return series, '等待首条日志：训练尚未写出 loss。', ''
    except OSError as exc:
        return series, '读取日志失败。', str(exc)
    last = max((max(s) for s in series.values() if s), default=None)
    status = f'最新记录：step {last}' if last is not None else '等待首条完整的指标记录。'
    notes = []
    if warnings:
        notes.append(f'发现 {len(warnings)} 个非有限或非数值指标（最近：{warnings[-1]}），不作为 0 绘制。')
    if malformed:
        notes.append(f'已忽略 {malformed} 条无效记录。')
    return series, status, ' '.join(notes)


def chart(key, samples, end):
    title, color = METRICS[key]
    heading = f'<h2>{title}</h2><code>{key}</code>'
    if not samples:
        return f'<article>{heading}<p>暂无记录</p></article>'
    last = max(samples)
    val = samples[last]
    value = f'{val:.5g}' if val is not None else '非有限或无效值'
    caption = f'<p><b class="value">{value}</b> <small>· step {last}</small></p>'
    good = [v for v in samples.values() if v is not None]
    if not good:
        return f'<article>{heading}{caption}</article>'
    lo, hi = min(good), max(good)
    pad = ((hi - lo) or abs(hi) or 1.) * .07
    low, high = max(0., lo - pad) if lo >= 0 else lo - pad, hi + pad
    svg = [f'<svg viewBox="0 0 460 150" role="img" aria-label="{title}随训练步数的变化">']
    for frac in (0, .5, 1):
        y = 12 + 105 * frac
        tick = high - (high - low) * frac
        svg.append(f'<path d="M71 {y} H445" stroke="#dbe3e8"/><text x="66" y="{y+4}" text-anchor="end">{tick:.3g}</text>')
    svg.append('<text x="71" y="139">0</text><text x="258" y="139" text-anchor="middle">训练步数</text>')
    if end:
        svg.append(f'<text x="445" y="139" text-anchor="end">{end}</text>')
    path, dots, pen = [], [], 'M'
    for step, v in sorted(samples.items()):
        if v is None:
            pen = 'M'
            continue
        x, y = 71 + 374 * step / max(end, 1), 117 - 105 * (v - low) / (high - low)
        path.append(f'{pen}{x:.2f} {y:.2f}')
        pen = 'L'
        dots.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5"><title>step {step}: {v:.6g}</title></circle>')
    svg.extend([f'<path d="{" ".join(path)}" stroke="{color}" stroke-width="2" fill="none"/>',
                f'<g fill="{color}">{"".join(dots)}</g></svg>'])
    return f'<article>{heading}{caption}{"".join(svg)}</article>'


def page(route, log):
    series, status, note = snapshot(log)
    end = max((max(s) for s in series.values() if s), default=0)
    nav = ' '.join(f'<a href="{r}"' + (' aria-current="page"' if r == route else '') + f'>{label}</a>'
                   for r, label in [('/', 'Smoke 测试'), ('/train', '正式训练')])
    cards = ''.join(chart(k, samples, end) for k, samples in series.items())
    warning = f'<p class="warn">{escape(note)}</p>' if note else ''
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="3">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>DreamZero · Loss</title><style>{STYLE}</style></head>
<body><main><h1>翻枕头 · 训练指标</h1><nav>{nav}</nav><p><strong>{escape(status)}</strong> · 每 3 秒刷新</p>{warning}
<div class="grid">{cards}</div><footer>只读日志：<code>{escape(str(log))}</code><br>分项指标约每 10 步记录一次。10 步测试中只有一个动作损失点也属正常。</footer></main></body></html>'''


def make_app(logs):
    def app(environ, start_response):
        route, method = environ.get('PATH_INFO', '/'), environ['REQUEST_METHOD']
        headers = [('Cache-Control', 'no-store'), ('X-Content-Type-Options', 'nosniff'),
                   ('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'")]
        if method not in ('GET', 'HEAD'):
            status, content = '405 Method Not Allowed', 'Read only.'
            headers.append(('Allow', 'GET, HEAD'))
        elif route not in logs:
            status, content = '404 Not Found', 'Not found.'
        else:
            status, content = '200 OK', page(route, logs[route])
        data = content.encode('utf-8')
        start_response(status, headers + [('Content-Type', 'text/html; charset=utf-8'), ('Content-Length', str(len(data)))])
        return [] if method == 'HEAD' else [data]
    return app


class QuietHandler(WSGIRequestHandler):
    def log_request(self, code='-', size='-'):
        if str(code) != '200':
            super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('smoke', 'train'):
        parser.add_argument(f'--{name}-dir', type=Path, default=ROOT / f'checkpoints/flip_pillow_lora_{name}',
                            help=f'{name} 模式的训练输出目录（不要求已存在）')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port 必须介于 1 和 65535 之间')
    logs = {route: folder.expanduser().resolve() / 'loss_log.jsonl' for route, folder in
            [('/', args.smoke_dir), ('/train', args.train_dir)]}
    try:
        with make_server('127.0.0.1', args.port, make_app(logs), handler_class=QuietHandler) as server:
            print(f'网页已启动：http://127.0.0.1:{args.port}/ （每 3 秒刷新；Ctrl+C 只停止网页）', flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        parser.exit(1, f'网页未启动：{exc}。若端口被占用，请用 --port 更换端口并同步调整 SSH 转发。\n')


if __name__ == '__main__':
    main()
