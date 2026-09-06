"""Render the README terminal demos as GIFs.

These are frame-by-frame renders of real sessions captured while building
the project (the outputs are verbatim), drawn with Pillow so the repo needs
no screen-recording toolchain to regenerate them:

    python docs/assets/render_demos.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

W, H = 900, 440
PAD, LINE = 22, 22
BG, FG = (24, 26, 32), (214, 218, 226)
DIM, GREEN, YELLOW, RED, CYAN, MAG = ((120, 126, 138), (129, 200, 140),
                                      (230, 190, 100), (235, 110, 110),
                                      (110, 190, 220), (200, 140, 220))

Line = tuple[str, tuple[int, int, int]] | tuple[str, tuple[int, int, int], bool]


def _fonts():
    return ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT_B, 15)


def frame(title: str, lines: list[Line], cursor: bool = False) -> Image.Image:
    font, bold = _fonts()
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # window chrome
    d.rounded_rectangle((0, 0, W, 34), radius=0, fill=(36, 39, 47))
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse((14 + i * 22, 10, 28 + i * 22, 24), fill=c)
    d.text((W // 2 - len(title) * 4, 9), title, font=font, fill=DIM)
    y = 34 + PAD // 2
    for entry in lines[-((H - 50) // LINE):]:
        text, color = entry[0], entry[1]
        use_bold = len(entry) > 2 and entry[2]
        d.text((PAD, y), text, font=bold if use_bold else font, fill=color)
        y += LINE
    if cursor:
        d.rectangle((PAD, y + 3, PAD + 9, y + 18), fill=FG)
    return img


def save(name: str, frames: list[tuple[Image.Image, int]]) -> None:
    images = [f for f, _ in frames]
    durations = [ms for _, ms in frames]
    images[0].save(HERE / name, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    print("wrote", name, len(frames), "frames")


def typed(prompt_lines: list[Line], command: str, color=FG):
    """Yield frames of a command being typed."""
    out = []
    for i in range(1, len(command) + 1):
        out.append((frame("agentapi", prompt_lines + [("$ " + command[:i], color)],
                          cursor=True), 28))
    out.append((frame("agentapi", prompt_lines + [("$ " + command, color)]), 350))
    return out


# --------------------------------------------------------------------------
def disconnect_survival() -> None:
    frames = []
    base: list[Line] = []
    cmd = "curl -N localhost:8000/research -d '{\"topic\": \"event logs\"}'"
    frames += typed(base, cmd)
    shown = base + [("$ " + cmd, FG)]
    stream = [
        ("event: state_delta", MAG),
        ('data: {"stage": "gathered", "passages": [...]}', DIM),
        ("event: token   data: \"Event \"", GREEN),
        ("event: token   data: \"logs \"", GREEN),
        ("event: token   data: \"make \"", GREEN),
        ("event: token   data: \"streams \"", GREEN),
    ]
    for line in stream:
        shown = shown + [line]
        frames.append((frame("agentapi", shown), 260))
    shown = shown + [("^C", RED, True), ("", FG),
                     ("# client hung up mid-stream — the run is still going", DIM)]
    frames.append((frame("agentapi", shown), 1600))

    cmd2 = "curl localhost:8000/runs/run_c94e4b0f/events?stream=false | jq .events[].type"
    frames += typed(shown, cmd2, CYAN)
    shown = shown + [("$ " + cmd2, CYAN)]
    result = [("state_delta", MAG)] + [("token", GREEN)] * 6 + [
        ("token   ← generated after the client left", GREEN),
        ("token", GREEN), ("done    ← run completed anyway", YELLOW, True)]
    for line in result:
        shown = shown + [line]
        frames.append((frame("agentapi", shown), 110))
    shown = shown + [("", FG),
                     ("41 events. Nothing lost. Reattach any time with ?from=<seq>.", YELLOW, True)]
    frames.append((frame("agentapi", shown), 3500))
    save("disconnect-survival.gif", frames)


def approval() -> None:
    frames = []
    cmd = "curl -N localhost:8000/agent -d '{\"task\": \"create hello.txt\"}'"
    frames += typed([], cmd)
    shown: list[Line] = [("$ " + cmd, FG)]
    stream = [
        ("event: state_delta   {\"task\": \"create hello.txt\", \"tools\": [...]}", MAG),
        ("event: message       \"I'll write the file.\"", FG),
        ("event: tool_call     write_file {\"path\": \"hello.txt\", ...}", CYAN),
        ("event: state_delta   {\"awaiting_approval\": {\"tool\": \"write_file\"}}", YELLOW),
        ("event: paused        signal=approval", YELLOW, True),
    ]
    for line in stream:
        shown = shown + [line]
        frames.append((frame("agentapi", shown), 320))
    shown = shown + [("", FG), ("# policy says write_file needs a human. The run waits —", DIM),
                     ("# and would survive a process restart while it does.", DIM)]
    frames.append((frame("agentapi", shown), 1800))

    cmd2 = "curl -X POST localhost:8000/runs/run_cfaefe81/signals/approval -d '{\"approved\": true}'"
    frames += typed(shown, cmd2, GREEN)
    shown = shown + [("$ " + cmd2, GREEN),
                     ('{"ok": true, "signal": "approval"}', DIM)]
    frames.append((frame("agentapi", shown), 500))
    resumed = [
        ("event: resumed       payload={\"approved\": true}", GREEN),
        ("event: tool_result   {\"path\": \"hello.txt\", \"bytes\": 68}", CYAN),
        ("event: message       \"Done: hello.txt now holds a haiku.\"", FG),
        ("event: done          usage: $0.0004 · 1 tool call", YELLOW, True),
    ]
    for line in resumed:
        shown = shown + [line]
        frames.append((frame("agentapi", shown), 320))
    frames.append((frame("agentapi", shown), 3200))
    save("human-approval.gif", frames)


def crash_recovery() -> None:
    frames = []
    shown: list[Line] = [
        ("$ uvicorn app:app --port 8000        # worker A", DIM),
        ("$ curl -N localhost:8000/pipeline -d '{\"topic\": \"kv\"}'", FG),
        ("event: state_delta   {\"gathered\": \"gathered:kv\"}   ← expensive step ran", MAG),
        ("event: paused        signal=approval", YELLOW, True),
    ]
    frames.append((frame("agentapi", shown), 1400))
    shown = shown + [("", FG)]
    cmd = "kill -9 $(pgrep -f 'worker A')     # simulate a crash / bad deploy"
    frames += typed(shown, cmd, RED)
    shown = shown + [("$ " + cmd, RED), ("Killed", RED, True), ("", FG)]
    frames.append((frame("agentapi", shown), 1400))

    cmd2 = "uvicorn app:app --port 8000        # worker B, same journal"
    frames += typed(shown, cmd2, DIM)
    shown = shown + [("$ " + cmd2, DIM),
                     ("[recover] claimed run_bc3a8a2c from journal", CYAN),
                     ("[recover] replaying 2 events, 1 journaled step, 0 signals", CYAN),
                     ("[recover] step 'expensive' → journaled result, NOT re-executed", GREEN, True),
                     ("[recover] run_bc3a8a2c is paused again, awaiting 'approval'", YELLOW)]
    for i in range(1, 5):
        frames.append((frame("agentapi", shown[:-4 + i] if i < 4 else shown), 420))
    frames.append((frame("agentapi", shown), 1200))

    cmd3 = "curl -X POST localhost:8000/runs/run_bc3a8a2c/signals/approval -d '{\"approved\": true}'"
    frames += typed(shown, cmd3, GREEN)
    shown = shown + [("$ " + cmd3, GREEN),
                     ("event: resumed", GREEN),
                     ("event: token  ×4", GREEN),
                     ("event: done   result={\"approved\": true}", YELLOW, True),
                     ("", FG),
                     ("Side effects ran once. History has no duplicates. Zero code changes.", YELLOW, True)]
    for i in range(1, 7):
        frames.append((frame("agentapi", shown[:-6 + i] if i < 6 else shown), 300))
    frames.append((frame("agentapi", shown), 3600))
    save("crash-recovery.gif", frames)


if __name__ == "__main__":
    disconnect_survival()
    approval()
    crash_recovery()
