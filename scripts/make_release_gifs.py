#!/usr/bin/env python3
"""Render the animated release GIFs in docs/assets/ from measured speedy rc2 data.

Usage: python3 scripts/make_release_gifs.py [--out docs/assets]

All numbers are measurements, not targets. Sources:
- RESTORE: .omo-work/restore-500k-2026-11-18/restore.log (502K-token prompt)
- DECODE:  .omo-work/bench-2026-11-18/quick-rc2-nvfp4 (BetterBench, NVFP4 KV)
- FIELD:   Prometheus sglang:* counters, 30 h of agent production up to 2026-11-19 06:05 CET; tok/s = median
           sglang:gen_throughput while 8 requests run
- COMPARE: README and launch scripts of jpezzulli/sglang-rtxpro6000 (3.1.6) and
           hyperburst-rtxpro6000 (2026-11-12); speedy column from examples/run-rc2-production.sh
- HOSTRAM: HyperBurst README "Host RAM" section (100.5 GiB default peak, 53.6 GiB with HiCache off);
           speedy = memory.current of the qwen38-rc2 cgroup under full agent load, 2026-11-19 05:30 CET
- STARTUP: server_args -> "fired up" in .omo-work/*/startup*.log and *.server.log, medians of
           5 runs before (2026-11-17/18) and 6 runs with persistent caches (rc2, 2026-11-18)
"""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESTORE = {"tokens": 502_000, "cold_s": 38.3, "cached_s": 1.4}
DECODE = [(1, 210), (2, 346), (4, 489), (8, 618)]
FIELD = {
    "requests": 35_720,
    "prompt_tokens": 2_310_000_000,
    "cache_pct": 94.9,
    "generated": 26_600_000,
    "tps_8_agents": 642,
}
W, H = 880, 420
BG = (11, 14, 26)
GRID = (30, 36, 58)
FG = (230, 235, 255)
DIM = (120, 130, 165)
CYAN = (0, 229, 255)
MAGENTA = (255, 46, 151)
LIME = (166, 255, 0)
AMBER = (255, 184, 0)
FPS = 25

HOSTRAM = [
    ("HyperBurst, defaults", 100.5, MAGENTA, "HiCache 32 GB"),
    ("HyperBurst, HiCache off", 53.6, AMBER, "no prefix cache in RAM"),
    ("speedy rc2", 34.8, LIME, "4 GB buffer, cache on NVMe"),
]
STARTUP = {"before_s": 359, "after_s": 281, "autotune_before_s": 52, "autotune_after_s": 4}
HOST = "ThinkPad L15 Gen 2  |  i5-1145G7 (4C/8T)  |  64 GB RAM  |  RTX PRO 6000 over USB4 eGPU"
UNIQUE = [
    ("NVIDIA's official checkpoint", "Qwen3.8-Flash-Next-FP4-Local, packed NVFP4 PLE (3 open NVIDIA PRs)"),
    ("Runs on a 64 GB host", "PLE pinned in RAM: 26.8 GiB instead of 47.7 GiB (FP8)"),
    ("Buffer-only HiCache on NVMe", "no RAM cache: 4 GB buffer, 200 GB SSD (saves a lot of host RAM!)"),
    ("NVFP4 KV + HiCache", "FP4 block scales travel with the KV (the overlay refuses it)"),
    ("Self-learning draft vocab", "learning 8K map, LRU/LFU/tiered (vLLM V3 had a 1K ring)"),
    ("Faster startup", "359 s -> 281 s, autotune 52 s -> 4 s"),
]

FONT_DIR = Path("/usr/share/fonts")


def font(size, bold=True):
    names = ["NotoSansMono-Bold.ttf", "DejaVuSansMono-Bold.ttf"] if bold else ["NotoSansMono-Regular.ttf", "DejaVuSansMono.ttf"]
    for name in names:
        hits = list(FONT_DIR.rglob(name))
        if hits:
            return ImageFont.truetype(str(hits[0]), size)
    return ImageFont.load_default(size)


F_TITLE, F_BIG, F_MID, F_SMALL = font(26), font(44), font(22), font(15, bold=False)


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def canvas(title, subtitle):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    for x in range(0, W, 40):
        d.line([(x, 0), (x, H)], fill=GRID)
    for y in range(0, H, 40):
        d.line([(0, y), (W, y)], fill=GRID)
    d.rectangle([0, 0, W, 4], fill=CYAN)
    d.text((28, 20), title, font=F_TITLE, fill=FG)
    d.text((28, 56), subtitle, font=F_SMALL, fill=DIM)
    d.text((W - 28, H - 26), "speedy-luni-sglang-rtx6000 rc2", font=F_SMALL, fill=DIM, anchor="ra")
    return im, d


def glow_bar(d, x0, y0, x1, y1, color):
    for i, a in enumerate((0.18, 0.32)):
        pad = 6 - 3 * i
        c = tuple(int(BG[k] + (color[k] - BG[k]) * a) for k in range(3))
        d.rounded_rectangle([x0 - pad, y0 - pad, max(x0, x1) + pad, y1 + pad], radius=10, fill=c)
    if x1 > x0:
        d.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=color)


def save(frames, path, hold=40):
    frames = frames + [frames[-1]] * hold
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 / FPS), loop=0, optimize=True)


def restore_race(out):
    """502K-token prompt: cold prefill vs prefix-cache hit, wall clock compressed 10x."""
    frames = []
    speed = 10.0
    total = int(RESTORE["cold_s"] / speed * FPS) + 8
    x0, x1 = 230, W - 60
    for f in range(total):
        sim = f / FPS * speed
        im, d = canvas(
            "Half a million tokens of context",
            f"{RESTORE['tokens']:,}-token prompt on one RTX PRO 6000  |  playback 10x real time",
        )
        d.text((W - 28, 22), f"t = {min(sim, RESTORE['cold_s']):5.1f} s", font=F_MID, fill=AMBER, anchor="ra")
        rows = [("cold", RESTORE["cold_s"], MAGENTA, 130), ("cached", RESTORE["cached_s"], LIME, 250)]
        for label, dur, col, y in rows:
            p = min(1.0, sim / dur)
            d.text((28, y + 8), label, font=F_MID, fill=col)
            d.rounded_rectangle([x0, y, x1, y + 56], radius=8, outline=GRID, width=2)
            glow_bar(d, x0, y, x0 + (x1 - x0) * p, y + 56, col)
            if p >= 1.0:
                d.text((x0 + 16, y + 10), f"done in {dur:.1f} s", font=F_MID, fill=BG)
            else:
                d.text((x0 + 16 + (x1 - x0) * p, y + 14), f"{p * 100:4.0f} %", font=F_SMALL, fill=FG)
        if sim >= RESTORE["cold_s"]:
            d.text((W // 2, 345), f"{RESTORE['cold_s'] / RESTORE['cached_s']:.0f}x faster from prefix cache", font=F_MID, fill=CYAN, anchor="ma")
        frames.append(im)
    save(frames, out / "speedy-restore-race.gif", hold=60)


def decode_scaling(out):
    """Aggregate decode throughput per concurrent stream count (NVFP4 KV)."""
    frames = []
    peak = DECODE[-1][1]
    n = 60
    base_y, top_y = 350, 110
    bw, gap, left = 130, 60, 120
    for f in range(n):
        im, d = canvas("Decode scales with your agent swarm", "aggregate tok/s, NVFP4 KV, adaptive MTP  |  BetterBench quick")
        for i, (streams, tps) in enumerate(DECODE):
            t = ease((f - i * 8) / 28)
            h = (base_y - top_y) * tps / peak * t
            x = left + i * (bw + gap)
            col = (CYAN, CYAN, MAGENTA, LIME)[i]
            glow_bar(d, x, base_y - h, x + bw, base_y, col)
            if h > 2:
                d.rounded_rectangle([x, base_y - h, x + bw, base_y], radius=8, fill=col)
            d.text((x + bw / 2, base_y - h - 34), f"{int(tps * t)}", font=F_MID, fill=FG, anchor="ma")
            d.text((x + bw / 2, base_y + 12), f"{streams} stream{'s' if streams > 1 else ''}", font=F_SMALL, fill=DIM, anchor="ma")
        frames.append(im)
    save(frames, out / "speedy-decode-scaling.gif")


def human(v):
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= div:
            return f"{v / div:.2f}{suf}"
    return f"{v:.0f}"


def field_counters(out):
    """30 h of real coding-agent production, counters roll up from zero."""
    frames = []
    n = 70
    tiles = [
        ("requests", FIELD["requests"], lambda v: f"{int(v):,}", CYAN),
        ("prompt tokens", FIELD["prompt_tokens"], human, MAGENTA),
        ("from prefix cache", FIELD["cache_pct"], lambda v: f"{v:.1f} %", LIME),
        ("tok/s, 8 agents (median)", FIELD["tps_8_agents"], lambda v: f"{int(v):,}", AMBER),
    ]
    for f in range(n):
        im, d = canvas("30 hours of real agent production", "one RTX PRO 6000  |  concurrent coding agents  |  Prometheus counters")
        for i, (label, val, fmt, col) in enumerate(tiles):
            t = ease((f - i * 5) / 40)
            cx = 28 + (i % 2) * 420
            cy = 100 + (i // 2) * 140
            d.rounded_rectangle([cx, cy, cx + 400, cy + 120], radius=14, outline=col, width=3)
            d.text((cx + 22, cy + 16), fmt(val * t), font=F_BIG, fill=col)
            d.text((cx + 22, cy + 80), label, font=F_SMALL, fill=DIM)
            if label == "from prefix cache":
                d.rounded_rectangle([cx + 250, cy + 84, cx + 380, cy + 98], radius=6, outline=GRID, width=1)
                d.rounded_rectangle([cx + 250, cy + 84, cx + 250 + 130 * val / 100 * t, cy + 98], radius=6, fill=col)
        pulse = 0.5 + 0.5 * math.sin(f / 4)
        c = tuple(int(DIM[k] + (CYAN[k] - DIM[k]) * pulse) for k in range(3))
        d.ellipse([W - 230, 30, W - 218, 42], fill=c)
        d.text((W - 210, 26), "LIVE DATA", font=F_SMALL, fill=c)
        frames.append(im)
    save(frames, out / "speedy-field-30h.gif")


def compare_table(out):
    """Feature matrix, rows type in one by one, speedy column lights up at the end."""
    cw, ch = 1180, 690
    cols = ["", "SGLang nightly", "Pennyroyal 3.1.6", "HyperBurst", "speedy rc2"]
    xs = [24, 220, 430, 660, 880]
    frames = []
    n_rows = len(COMPARE)
    per = 9
    total = n_rows * per + 30
    for f in range(total):
        im = Image.new("RGB", (cw, ch), BG)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, cw, 4], fill=CYAN)
        d.text((24, 18), "What speedy adds", font=F_TITLE, fill=FG)
        d.text((24, 54), "Qwen3.8 Flash-Next on one RTX PRO 6000 Blackwell  |  from each project's README and launch configs", font=F_SMALL, fill=DIM)
        hl = ease((f - n_rows * per) / 20)
        if hl > 0:
            c = tuple(int(BG[k] + (CYAN[k] - BG[k]) * 0.22 * hl) for k in range(3))
            d.rounded_rectangle([xs[4] - 12, 88, cw - 12, ch - 40], radius=14, fill=c, outline=CYAN, width=2)
        for i, name in enumerate(cols):
            d.text((xs[i], 98), name, font=F_SMALL if i < 4 else font(16), fill=CYAN if i == 4 else FG)
        d.line([(24, 124), (cw - 24, 124)], fill=GRID, width=2)
        for r, row in enumerate(COMPARE):
            shown = (f - r * per) / per
            if shown <= 0:
                break
            y = 136 + r * 44
            for i, cell in enumerate(row):
                text = cell[: max(0, int(len(cell) * min(1.0, shown * 1.6)))]
                if i == 0:
                    col = DIM
                elif i == 4:
                    col = LIME
                else:
                    col = FG
                d.text((xs[i], y), text, font=font(15, bold=i in (0, 4)), fill=col)
        d.text((cw - 24, ch - 26), "speedy-luni-sglang-rtx6000 rc2", font=F_SMALL, fill=DIM, anchor="ra")
        frames.append(im)
    save(frames, out / "speedy-compare.gif", hold=75)


def only_in_speedy(out):
    """Brochure card: the six things no other Qwen3.8 Flash-Next SGLang build combines."""
    cw, ch = 980, 560
    frames = []
    per = 14
    total = len(UNIQUE) * per + 30
    for f in range(total):
        im = Image.new("RGB", (cw, ch), BG)
        d = ImageDraw.Draw(im)
        for x in range(0, cw, 40):
            d.line([(x, 0), (x, ch)], fill=GRID)
        d.rectangle([0, 0, cw, 4], fill=LIME)
        d.text((28, 18), "Only in speedy", font=F_TITLE, fill=FG)
        d.text((28, 54), "max SGLang speed for Qwen3.8 Flash-Next on a cheap 64 GB host + eGPU", font=F_SMALL, fill=DIM)
        for i, (head, sub) in enumerate(UNIQUE):
            t = ease((f - i * per) / 12)
            if t <= 0:
                break
            y = 92 + i * 72
            x = 28 + int((1 - t) * 60)
            col = tuple(int(BG[k] + (LIME[k] - BG[k]) * t) for k in range(3))
            d.rounded_rectangle([x, y, cw - 28, y + 60], radius=12, outline=col, width=2)
            d.ellipse([x + 14, y + 16, x + 42, y + 44], fill=col)
            d.line([(x + 20, y + 30), (x + 27, y + 38), (x + 37, y + 22)], fill=BG, width=4)
            d.text((x + 58, y + 7), head, font=font(20), fill=tuple(int(BG[k] + (FG[k] - BG[k]) * t) for k in range(3)))
            d.text((x + 58, y + 35), sub, font=F_SMALL, fill=tuple(int(BG[k] + (DIM[k] - BG[k]) * t) for k in range(3)))
        d.text((28, ch - 26), HOST, font=F_SMALL, fill=CYAN)
        frames.append(im)
    save(frames, out / "speedy-only-in-speedy.gif", hold=80)


def host_ram(out):
    """Host RAM of the serving process: HyperBurst (its README) vs speedy (live cgroup)."""
    frames = []
    n = 70
    x0, x1, limit, scale_max = 300, W - 40, 64e9 / 2**30, 110.0
    for f in range(n):
        im, d = canvas("Full speed on a 64 GB host", "host RAM of the server process, GiB  |  HyperBurst from its README, speedy measured live")
        lx = x0 + (x1 - x0) * limit / scale_max
        for i, (label, gib, col, note) in enumerate(HOSTRAM):
            t = ease((f - i * 10) / 30)
            y = 110 + i * 82
            d.text((28, y + 4), label, font=font(17), fill=col)
            d.text((28, y + 30), note, font=F_SMALL, fill=DIM)
            glow_bar(d, x0, y, x0 + (x1 - x0) * gib / scale_max * t, y + 44, col)
            d.text((x0 + 12, y + 10), f"{gib * t:5.1f} GiB", font=F_MID, fill=BG if t > 0.4 else FG)
        d.line([(lx, 92), (lx, 352)], fill=FG, width=3)
        d.text((lx, 358), "64 GB RAM = 59.6 GiB", font=F_SMALL, fill=FG, anchor="ma")
        d.text((28, H - 26), "ThinkPad L15 Gen 2 | i5-1145G7 | USB4 eGPU", font=F_SMALL, fill=CYAN)
        frames.append(im)
    save(frames, out / "speedy-64gb-host.gif", hold=60)


def startup(out):
    """Server start to ready: before vs with persistent kernel/autotune caches (medians)."""
    frames = []
    speed = 30.0
    total = int(STARTUP["before_s"] / speed * FPS) + 10
    x0, x1 = 230, W - 60
    for f in range(total):
        sim = f / FPS * speed
        im, d = canvas("Ready sooner after every restart", "server start to ready, median of real starts  |  playback 30x real time")
        d.text((W - 28, 22), f"t = {min(sim, STARTUP['before_s']):4.0f} s", font=F_MID, fill=AMBER, anchor="ra")
        rows = [("before", STARTUP["before_s"], MAGENTA, 130), ("speedy", STARTUP["after_s"], LIME, 230)]
        for label, dur, col, y in rows:
            p = min(1.0, sim / dur)
            d.text((28, y + 8), label, font=F_MID, fill=col)
            d.rounded_rectangle([x0, y, x1, y + 56], radius=8, outline=GRID, width=2)
            glow_bar(d, x0, y, x0 + (x1 - x0) * p * dur / STARTUP["before_s"], y + 56, col)
            if p >= 1.0:
                d.text((x0 + 16, y + 12), f"ready in {dur} s", font=F_MID, fill=BG)
        if sim >= STARTUP["before_s"]:
            saved = STARTUP["before_s"] - STARTUP["after_s"]
            d.text((W // 2, 315), f"-{saved} s (-{saved / STARTUP['before_s'] * 100:.0f} %)   autotune {STARTUP['autotune_before_s']} s -> {STARTUP['autotune_after_s']} s", font=F_MID, fill=CYAN, anchor="ma")
            d.text((W // 2, 350), "JIT kernels + autotune tactics persist  |  packed NVFP4 PLE 26.8 GiB instead of 47.7", font=F_SMALL, fill=FG, anchor="ma")
        frames.append(im)
    save(frames, out / "speedy-startup.gif", hold=60)


COMPARE = [
    ("Base", "current nightly", "SGLang 0.5.19 line", "Pennyroyal + 121", "nightly 11-17 + 154"),
    ("Built for", "general serving", "long chats, TP1/TP2", "1-stream decode", "agent swarms"),
    ("Context", "stock", "524K", "4 x 262K", "512K, 1M shared pool"),
    ("Concurrency", "stock", "1-6 requests", "up to 4", "8 running"),
    ("Checkpoint", "3 NVIDIA PRs open", "OrcaRouter, RadixArk", "RadixArk NVFP4", "NVIDIA FP4-Local"),
    ("Host RAM", "-", "~48 GiB PLE alone", "54-100 GiB", "34.8 GiB, 64 GB host"),
    ("KV cache", "FP8 / BF16", "FP8", "FP8", "NVFP4 + HiCache"),
    ("Host cache", "HiCache", "32 GB RAM + NIXL disk", "32 GB RAM", "4 GB buffer + 200 GB NVMe"),
    ("Drafting", "NEXTN MTP", "MTP + FR-Spec", "MTP + NGRAM, OEA", "MTP, self-learning vocab"),
    ("Startup", "-", "-", "-", "281 s (-22 %)"),
    ("Field proof", "CI", "releases, community", "benchmarks", "30 h prod, 2.1B tokens"),
    ("Patch log", "git", "CHANGES, PROVENANCE", "patch series", "214 numbered QFN-NNNN"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs" / "assets"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    restore_race(out)
    decode_scaling(out)
    field_counters(out)
    compare_table(out)
    only_in_speedy(out)
    host_ram(out)
    startup(out)
    for p in sorted(out.glob("speedy-*.gif")):
        print(f"{p}  {p.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
