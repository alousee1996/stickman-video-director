import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import imageio_ffmpeg

W, H = 720, 1280
FPS = 24
TOTAL_FRAMES = 240  # 10.0 seconds


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def make_sky_background(w, h):
    # Deep indigo gradient
    y = np.linspace(0, 1, h)[:, None]
    r = (11 + y * 18).astype(np.uint8)
    g = (14 + y * 24).astype(np.uint8)
    b = (38 + y * 52).astype(np.uint8)
    arr = np.dstack([
        np.repeat(r, w, axis=1),
        np.repeat(g, w, axis=1),
        np.repeat(b, w, axis=1),
    ])
    return Image.fromarray(arr, "RGB")


def draw_moon(draw, cx, cy, r):
    # Soft volumetric glow
    for gr, alpha in [(r * 2.4, 18), (r * 1.7, 32), (r * 1.25, 55)]:
        draw.ellipse([cx - gr, cy - gr, cx + gr, cy + gr], fill=(220, 230, 255, alpha))
    # Pale moon disc
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(238, 242, 255, 255))


def draw_wooden_chair(draw, x, y, scale=1.0):
    # Modest wooden chair
    wood = (92, 62, 42, 255)
    wood_dark = (64, 42, 28, 255)
    lw = max(2, int(4 * scale))
    cw = 95 * scale
    ch = 110 * scale
    seat_y = y
    # Backrest posts
    draw.line([x - cw * 0.45, seat_y - ch, x - cw * 0.45, seat_y + ch * 0.65], fill=wood_dark, width=lw + 2)
    draw.line([x + cw * 0.45, seat_y - ch, x + cw * 0.45, seat_y + ch * 0.65], fill=wood, width=lw + 2)
    # Backrest slats
    for sy in [0.85, 0.55, 0.25]:
        draw.rectangle([x - cw * 0.45, seat_y - ch * sy, x + cw * 0.45, seat_y - ch * sy + 12 * scale], fill=wood)
    # Seat
    draw.rectangle([x - cw * 0.5, seat_y, x + cw * 0.5, seat_y + 14 * scale], fill=wood)
    # Front legs
    draw.line([x - cw * 0.42, seat_y + 12 * scale, x - cw * 0.42, seat_y + ch * 0.68], fill=wood, width=lw + 2)
    draw.line([x + cw * 0.42, seat_y + 12 * scale, x + cw * 0.42, seat_y + ch * 0.68], fill=wood_dark, width=lw + 2)


def draw_junior(draw, x, y, scale=1.0, head_tilt=-0.18, arm_lift=0.0, eye_dx=0.0, eye_dy=-2.0):
    """
    Teen Junior:
    - Bright red beanie (smooth rounded knit, no pom-pom)
    - Yellow t-shirt (#F5C518)
    - Long dark trousers (#1C2030)
    - Simple black stick limbs, constant line weight, minimal dot eyes
    """
    lw = max(3, int(6 * scale))
    black = (18, 20, 26, 255)
    silver_rim = (210, 224, 255, 160)
    yellow = (245, 197, 24, 255)
    red_beanie = (218, 42, 42, 255)
    red_cuff = (185, 30, 30, 255)
    trousers = (28, 32, 48, 255)

    # Torso & sitting posture
    hip_x, hip_y = x, y
    shoulder_x, shoulder_y = x - 6 * scale, y - 95 * scale

    # Long dark trousers (sitting: thigh forward, shin down)
    knee_x, knee_y = hip_x + 48 * scale, hip_y + 6 * scale
    foot_x, foot_y = knee_x + 8 * scale, knee_y + 72 * scale
    draw.line([hip_x, hip_y, knee_x, knee_y], fill=trousers, width=int(16 * scale))
    draw.line([knee_x, knee_y, foot_x, foot_y], fill=trousers, width=int(14 * scale))
    # Second leg slightly offset
    draw.line([hip_x - 6 * scale, hip_y + 2 * scale, knee_x - 10 * scale, knee_y + 4 * scale], fill=(22, 25, 38, 255), width=int(15 * scale))
    draw.line([knee_x - 10 * scale, knee_y + 4 * scale, foot_x - 12 * scale, foot_y], fill=(22, 25, 38, 255), width=int(13 * scale))

    # Yellow t-shirt torso
    torso_pts = [
        (shoulder_x - 22 * scale, shoulder_y),
        (shoulder_x + 24 * scale, shoulder_y),
        (hip_x + 22 * scale, hip_y + 6 * scale),
        (hip_x - 20 * scale, hip_y + 6 * scale),
    ]
    draw.polygon(torso_pts, fill=yellow, outline=black)
    # Soft silver moonlight rim on right side
    draw.line([shoulder_x + 24 * scale, shoulder_y, hip_x + 22 * scale, hip_y + 6 * scale], fill=silver_rim, width=max(2, int(3 * scale)))

    # Head (looking up)
    head_r = 28 * scale
    hx = shoulder_x + 4 * scale
    hy = shoulder_y - 34 * scale
    draw.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r], fill=(255, 255, 255, 255), outline=black, width=lw)
    # Silver rim on head
    draw.arc([hx - head_r, hy - head_r, hx + head_r, hy + head_r], start=-65, end=35, fill=silver_rim, width=max(2, int(3 * scale)))

    # Bright red beanie (smooth rounded knit, no pom-pom)
    beanie_box = [hx - head_r - 2 * scale, hy - head_r - 12 * scale, hx + head_r + 2 * scale, hy + 2 * scale]
    draw.chord(beanie_box, start=180, end=360, fill=red_beanie, outline=black, width=lw)
    # Beanie cuff
    draw.rounded_rectangle(
        [hx - head_r - 3 * scale, hy - 10 * scale, hx + head_r + 3 * scale, hy + 2 * scale],
        radius=5 * scale,
        fill=red_cuff,
        outline=black,
        width=max(2, int(lw * 0.8)),
    )

    # Minimal dot eyes (following sky)
    ex1 = hx + (8 + eye_dx) * scale
    ex2 = hx + (20 + eye_dx) * scale
    ey = hy + (2 + eye_dy) * scale
    er = max(2, 3.2 * scale)
    draw.ellipse([ex1 - er, ey - er, ex1 + er, ey + er], fill=black)
    draw.ellipse([ex2 - er, ey - er, ex2 + er, ey + er], fill=black)

    # Arms:
    # Left/back arm resting chin on hand (or lowered slightly)
    elbow1_x, elbow1_y = shoulder_x + 24 * scale, shoulder_y + 30 * scale
    hand1_x, hand1_y = hx + 10 * scale, hy + head_r - 2 * scale
    draw.line([shoulder_x + 12 * scale, shoulder_y + 8 * scale, elbow1_x, elbow1_y], fill=black, width=lw)
    draw.line([elbow1_x, elbow1_y, hand1_x, hand1_y], fill=black, width=lw)

    # Right/front arm: chin on hand when arm_lift==0, lifts toward sky when arm_lift>0
    if arm_lift <= 0.01:
        elbow2_x, elbow2_y = shoulder_x + 30 * scale, shoulder_y + 34 * scale
        hand2_x, hand2_y = hx + 16 * scale, hy + head_r
        draw.line([shoulder_x + 18 * scale, shoulder_y + 10 * scale, elbow2_x, elbow2_y], fill=black, width=lw)
        draw.line([elbow2_x, elbow2_y, hand2_x, hand2_y], fill=black, width=lw)
    else:
        # Lift hand toward upper right sky
        ang = lerp(0.4, -1.15, arm_lift)
        arm_len = 78 * scale
        hand2_x = shoulder_x + 16 * scale + math.cos(ang) * arm_len
        hand2_y = shoulder_y + 10 * scale + math.sin(ang) * arm_len
        elbow2_x = lerp(shoulder_x + 16 * scale, hand2_x, 0.5) + 8 * scale
        elbow2_y = lerp(shoulder_y + 10 * scale, hand2_y, 0.5) + 8 * scale
        draw.line([shoulder_x + 16 * scale, shoulder_y + 10 * scale, elbow2_x, elbow2_y], fill=black, width=lw)
        draw.line([elbow2_x, elbow2_y, hand2_x, hand2_y], fill=black, width=lw)


def draw_clock_face(draw, cx, cy, r, alpha):
    if alpha <= 0.01:
        return
    a = int(255 * alpha)
    # Soft outer glow
    draw.ellipse([cx - r * 1.15, cy - r * 1.15, cx + r * 1.15, cy + r * 1.15], fill=(255, 224, 130, int(45 * alpha)))
    # Clock disc
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(242, 244, 250, a), outline=(30, 36, 58, a), width=max(3, int(r * 0.04)))
    # 12 unmarked tick marks (no numbers, no letters)
    for i in range(12):
        ang = math.radians(i * 30 - 90)
        r_in = r * (0.80 if i % 3 == 0 else 0.86)
        r_out = r * 0.93
        x1, y1 = cx + math.cos(ang) * r_in, cy + math.sin(ang) * r_in
        x2, y2 = cx + math.cos(ang) * r_out, cy + math.sin(ang) * r_out
        draw.line([x1, y1, x2, y2], fill=(30, 36, 58, a), width=max(2, int(r * (0.03 if i % 3 == 0 else 0.018))))

    # Unmarked hands near seven o'clock:
    # Hour hand near 7 (210 deg from 12 -> 210 - 90 = 120 deg)
    hr_ang = math.radians(120)
    draw.line([cx, cy, cx + math.cos(hr_ang) * r * 0.50, cy + math.sin(hr_ang) * r * 0.50], fill=(25, 30, 48, a), width=max(4, int(r * 0.05)))
    # Minute hand near 12 (slightly before 12: -95 deg)
    min_ang = math.radians(-95)
    draw.line([cx, cy, cx + math.cos(min_ang) * r * 0.72, cy + math.sin(min_ang) * r * 0.72], fill=(25, 30, 48, a), width=max(3, int(r * 0.035)))
    # Center pin
    pr = max(3, r * 0.05)
    draw.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=(25, 30, 48, a))


def main():
    rng = np.random.default_rng(7)
    # Generate star field
    num_stars = 140
    stars = []
    for _ in range(num_stars):
        sx = rng.uniform(20, W - 20)
        sy = rng.uniform(20, H * 0.72)
        sr = rng.uniform(1.2, 3.2)
        phase = rng.uniform(0, 2 * math.pi)
        is_gold = rng.random() < 0.35
        stars.append((sx, sy, sr, phase, is_gold))

    base_sky = make_sky_background(W, H)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-i", "output/final_audio.wav",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "output/junior_balcony.mp4",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for f in range(TOTAL_FRAMES):
        t = f / FPS  # 0.0 to 10.0

        # Camera & shot choreography:
        # [0-3s]: Tight on Junior's upturned face (zoom=1.85, focus on Junior)
        # [3-7s]: Tilt up to reveal vast star field (pan sky down / camera tilts up)
        # [7-10s]: Dolly back to show Junior small beside empty chair, then clock fills frame
        if t < 3.0:
            cam_scale = 1.85
            cam_y_offset = -120.0
            cam_x_offset = 90.0
        elif t < 7.0:
            p = smoothstep((t - 3.0) / 4.0)
            cam_scale = lerp(1.85, 1.15, p)
            cam_y_offset = lerp(-120.0, 180.0, p)
            cam_x_offset = lerp(90.0, 0.0, p)
        else:
            p = smoothstep((t - 7.0) / 3.0)
            cam_scale = lerp(1.15, 0.85, p)
            cam_y_offset = lerp(180.0, 0.0, p)
            cam_x_offset = 0.0

        frame = base_sky.copy().convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")

        # Moon in upper left
        moon_y = 210 + cam_y_offset * 0.25
        draw_moon(draw, 160 + cam_x_offset * 0.2, moon_y, 58)

        # Twinkling stars
        for sx, sy, sr, phase, is_gold in stars:
            tw = 0.55 + 0.45 * math.sin(t * 2.8 + phase)
            col = (255, 224, 130, int(255 * tw)) if is_gold else (230, 238, 255, int(235 * tw))
            sy_p = sy + cam_y_offset * 0.35
            draw.ellipse([sx - sr, sy_p - sr, sx + sr, sy_p + sr], fill=col)

        # Shooting star streak between t=3.8s and t=5.2s
        if 3.8 <= t <= 5.5:
            sp = (t - 3.8) / 1.4
            sp_clamped = min(1.0, sp)
            sx0, sy0 = 120.0, 180.0 + cam_y_offset * 0.35
            sx1, sy1 = 580.0, 380.0 + cam_y_offset * 0.35
            cur_x = lerp(sx0, sx1, sp_clamped)
            cur_y = lerp(sy0, sy1, sp_clamped)
            tail_x = lerp(sx0, sx1, max(0.0, sp_clamped - 0.25))
            tail_y = lerp(sy0, sy1, max(0.0, sp_clamped - 0.25))
            alpha_s = int(255 * (1.0 - max(0.0, (t - 5.0) / 0.5)))
            draw.line([tail_x, tail_y, cur_x, cur_y], fill=(255, 224, 130, alpha_s), width=5)
            draw.ellipse([cur_x - 6, cur_y - 6, cur_x + 6, cur_y + 6], fill=(255, 245, 200, alpha_s))

        # Balcony floor & chairs & Junior
        balcony_y = 920 + cam_y_offset * 0.8
        # Two wooden chairs side by side
        chair_scale = cam_scale * 0.95
        left_chair_x = W * 0.36 + cam_x_offset
        right_chair_x = W * 0.64 + cam_x_offset

        draw_wooden_chair(draw, left_chair_x, balcony_y, scale=chair_scale)
        draw_wooden_chair(draw, right_chair_x, balcony_y, scale=chair_scale)

        # Junior sitting in left chair
        arm_lift = 0.0
        if 4.0 <= t <= 6.5:
            if t < 5.0:
                arm_lift = smoothstep((t - 4.0) / 1.0)
            else:
                arm_lift = 1.0 - smoothstep((t - 5.0) / 1.5)

        eye_dx = math.sin(t * 1.2) * 1.5
        draw_junior(
            draw,
            left_chair_x + 4 * chair_scale,
            balcony_y - 4 * chair_scale,
            scale=chair_scale,
            arm_lift=arm_lift,
            eye_dx=eye_dx,
            eye_dy=-3.0,
        )

        # Foreground wooden railing with soft depth-of-field blur
        railing_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        rdraw = ImageDraw.Draw(railing_layer, "RGBA")
        rail_y = H - 180 + cam_y_offset * 0.4
        rdraw.rectangle([0, rail_y, W, rail_y + 26], fill=(48, 32, 22, 230))
        for rx in range(40, W, 90):
            rdraw.rectangle([rx, rail_y + 26, rx + 18, H], fill=(38, 25, 18, 230))
        railing_layer = railing_layer.filter(ImageFilter.GaussianBlur(radius=4))
        frame = Image.alpha_composite(frame, railing_layer)

        # [7-10s]: Clock face fades in & grows to fill the frame on final frame
        if t >= 7.0:
            cp = smoothstep((t - 7.0) / 2.5)
            clock_r = lerp(90, 310, cp)
            clock_cy = lerp(380, H * 0.5, cp)
            cdraw = ImageDraw.Draw(frame, "RGBA")
            draw_clock_face(cdraw, W * 0.5, clock_cy, clock_r, alpha=min(1.0, cp * 1.15))

        proc.stdin.write(frame.convert("RGB").tobytes())

    proc.stdin.close()
    proc.wait()
    print("Rendered output/junior_balcony.mp4 successfully.")


if __name__ == "__main__":
    main()
