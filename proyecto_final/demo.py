import os
import time
import math
import numpy as np
import cv2

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RENDERS_DIR = os.path.join(BASE_DIR, "renders")
os.makedirs(RENDERS_DIR, exist_ok=True)

W, H = 800, 600
FPS = 30
DURATION = 60.0
SHOW_WINDOW = True  # Cambia a False si solo quieres exportar sin abrir ventana

# Utilidades
def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def smoothstep(a, b, x):
    x = clamp01((x - a) / (b - a))
    return x * x * (3 - 2 * x)

def poly_param(fx, fy, t0, t1, n, cx, cy, sx, sy):
    ts = np.linspace(t0, t1, n, dtype=np.float32)
    xs = fx(ts) * sx + cx
    ys = fy(ts) * sy + cy
    return np.round(np.stack([xs, ys], 1)).astype(np.int32).reshape((-1, 1, 2))

def hsv_to_bgr(h, s, v):
    hsv = np.uint8([[[h % 180, np.clip(s, 0, 255), np.clip(v, 0, 255)]]])
    return tuple(int(x) for x in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])



def draw_soft_circle(img, center, radius, color, alpha=0.35, blur=10):
    """Glow ligero: círculo suave sin saturar demasiado."""
    glow = np.zeros_like(img)
    cv2.circle(glow, center, radius, color, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), blur)
    cv2.addWeighted(img, 1.0, glow, alpha, 0, dst=img)


def draw_text_glow(img, text, pos, scale, color, thick=2, glow_color=(255, 255, 255)):
    """Texto con sombra/glow para que se vea más chido sin gastar mucho rendimiento."""
    x, y = pos
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, glow_color, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_light_nebula(img, t, seed=0, amount=8):
    """Nubes suaves de color, pocas para que no se trabe."""
    rng = np.random.default_rng(seed)
    layer = np.zeros_like(img)
    for i in range(amount):
        cx = int((rng.integers(60, W - 60) + 35 * math.sin(t * 0.25 + i)) % W)
        cy = int((rng.integers(50, H - 50) + 25 * math.cos(t * 0.22 + i)) % H)
        rad = int(rng.integers(45, 95))
        hue = int(rng.integers(0, 179) + t * 4 + i * 7)
        cv2.circle(layer, (cx, cy), rad, hsv_to_bgr(hue, 140, 120), -1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 18)
    cv2.addWeighted(img, 1.0, layer, 0.20, 0, dst=img)


def draw_twinkle_stars(img, t, seed=1, n=70):
    """Estrellas extra con parpadeo, pocas para cuidar FPS."""
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, W, n)
    ys = rng.integers(0, H, n)
    for i, (x, y) in enumerate(zip(xs, ys)):
        b = int(120 + 100 * (0.5 + 0.5 * math.sin(t * 2.0 + i * 0.7)))
        cv2.circle(img, (int(x), int(y)), 1, (b, b, b), -1, cv2.LINE_AA)


def post_vignette(img, strength=0.7):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    mask = np.clip(1.0 - strength * r2, 0.0, 1.0)
    return (img.astype(np.float32) * mask[..., None]).astype(np.uint8)

def post_scanlines(img, strength=0.22):
    out = img.astype(np.float32)
    y = np.arange(H, dtype=np.float32)
    m = 1.0 - strength * (0.5 + 0.5 * np.sin(2 * np.pi * y / 3.0))
    out *= m[:, None, None]
    return np.clip(out, 0, 255).astype(np.uint8)

def post_posterize(img, q=32):
    q = max(1, int(q))
    return ((img // q) * q).astype(np.uint8)

def background_hsv_gradient(img, t, hue0=10, hue1=140):
    hsv = np.zeros((H, W, 3), np.uint8)
    ys = np.linspace(0, 1, H, dtype=np.float32)
    hue = hue0 + (hue1 - hue0) * ys + 10 * np.sin(t * 0.4 + ys * 2.0)

    hsv[:, :, 0] = np.clip(hue, 0, 179).astype(np.uint8)[:, None]
    hsv[:, :, 1] = 200
    hsv[:, :, 2] = (40 + 120 * (1 - ys)).astype(np.uint8)[:, None]

    img[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Guardar máscaras para el reporte
def guardar_mascaras():
    # Máscara de viñeta
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    mask_v = np.clip(1.0 - 0.72 * r2, 0.0, 1.0)

    mask_v_img = (mask_v * 255).astype(np.uint8)
    mask_v_color = cv2.applyColorMap(mask_v_img, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(RENDERS_DIR, "mascara_vignette.png"), mask_v_color)

    # Máscara circular de la escena final
    mask_c = np.zeros((H, W), np.uint8)
    cv2.circle(mask_c, (W // 2, H // 2), 270, 255, -1)
    mask_c_color = cv2.applyColorMap(mask_c, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(RENDERS_DIR, "mascara_circular_final.png"), mask_c_color)

# Helpers visuales
def draw_stars(img, seed=1, n=400):
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, W, n)
    ys = rng.integers(0, H, n)
    bri = rng.integers(100, 255, n)

    for i in range(n):
        b = int(bri[i])
        img[ys[i], xs[i]] = (b, b, b)

def draw_planet(img, cx, cy, radius, hue, rings=False):
    draw_soft_circle(img, (cx, cy), radius + 18, hsv_to_bgr(hue, 150, 160), alpha=0.18, blur=12)
    cv2.circle(img, (cx, cy), radius, hsv_to_bgr(hue, 200, 220), -1, cv2.LINE_AA)
    # Bandas sencillas para que el planeta no se vea tan plano
    for k in range(-2, 3):
        yy = cy + int(k * radius * 0.22)
        cv2.ellipse(img, (cx, yy), (radius - 4, max(2, radius // 9)), 0, 0, 360, hsv_to_bgr(hue + k * 5, 120, 210), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius, hsv_to_bgr(hue, 150, 255), 2, cv2.LINE_AA)

    shadow = np.zeros((H, W, 3), np.uint8)
    cv2.circle(shadow, (cx + radius // 3, cy), radius, (0, 0, 0), -1, cv2.LINE_AA)

    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    img[mask > 0] = np.where(
        shadow[mask > 0] > 0,
        (img[mask > 0].astype(np.float32) * 0.35).astype(np.uint8),
        img[mask > 0]
    )

    if rings:
        cv2.ellipse(
            img,
            (cx, cy),
            (radius + 40, radius // 4),
            0,
            0,
            360,
            hsv_to_bgr(hue + 10, 180, 200),
            2,
            cv2.LINE_AA
        )

        cv2.ellipse(
            img,
            (cx, cy),
            (radius + 55, radius // 3 + 2),
            0,
            0,
            360,
            hsv_to_bgr(hue + 20, 160, 180),
            1,
            cv2.LINE_AA
        )

# Escena 0 - Intro
def scene_credits(img, t):
    background_hsv_gradient(img, t, hue0=160, hue1=120)
    draw_stars(img, seed=7, n=430)
    draw_light_nebula(img, t, seed=11, amount=6)
    draw_twinkle_stars(img, t, seed=71, n=45)
    img[:] = cv2.GaussianBlur(img, (0, 0), 0.45)

    def arq_x(th):
        return th * np.cos(th + t * 0.4)

    def arq_y(th):
        return th * np.sin(th + t * 0.4)

    pts = poly_param(arq_x, arq_y, 0, 5 * math.pi, 900, W // 2, H // 2, 14, 14)

    cv2.polylines(
        img,
        [pts],
        False,
        hsv_to_bgr(140, 180, 200),
        1,
        cv2.LINE_AA
    )

    alpha = clamp01(t / 2.0)
    overlay = img.copy()

    draw_text_glow(
        overlay,
        "COSMOS PROCEDURAL",
        (150, 250),
        1.6,
        (255, 255, 255),
        3,
        hsv_to_bgr(145, 100, 255)
    )

    cv2.putText(
        overlay,
        "OpenCV + Python + Matematicas",
        (180, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (220, 220, 220),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        overlay,
        "6 escenas / curvas / transformaciones",
        (195, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    img[:] = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

# Escena 1 - Portal de energía
def scene_orbitas(img, t):
    background_hsv_gradient(img, t, hue0=30, hue1=90)
    draw_stars(img, seed=21, n=230)
    draw_light_nebula(img, t, seed=22, amount=5)
    draw_twinkle_stars(img, t, seed=23, n=45)

    cx, cy = W // 2, H // 2

    # Líneas radiales del portal
    for i in range(36):
        ang = 2 * math.pi * i / 36 + t * 0.25
        r1 = 70
        r2 = 330 + 20 * math.sin(t + i)

        x1 = int(cx + r1 * math.cos(ang))
        y1 = int(cy + r1 * math.sin(ang))
        x2 = int(cx + r2 * math.cos(ang))
        y2 = int(cy + r2 * math.sin(ang))

        col = hsv_to_bgr(45 + i * 2, 180, 160)
        cv2.line(img, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)

    # Curva epitrocoide
    R = 5.0
    r = 2.0
    d = 4.5
    phase = t * 0.8

    def ex(th):
        return (R + r) * np.cos(th) - d * np.cos(((R + r) / r) * th + phase)

    def ey(th):
        return (R + r) * np.sin(th) - d * np.sin(((R + r) / r) * th + phase)

    pts = poly_param(ex, ey, 0, 8 * math.pi, 1600, cx, cy, 28, 28)

    cv2.polylines(
        img,
        [pts],
        False,
        hsv_to_bgr(int(35 + 25 * math.sin(t)), 220, 245),
        2,
        cv2.LINE_AA
    )

    # Aros del portal
    for i in range(6):
        radius = int(70 + i * 35 + 8 * math.sin(t * 2 + i))
        color = hsv_to_bgr(35 + i * 8, 180, 210)
        cv2.circle(img, (cx, cy), radius, color, 1, cv2.LINE_AA)

    # Partículas alrededor del portal
    for p in range(28):
        a = t * 1.4 + p * 0.62
        rr = 95 + 145 * ((p % 7) / 7.0) + 8 * math.sin(t + p)
        px = int(cx + rr * math.cos(a))
        py = int(cy + rr * math.sin(a))
        cv2.circle(img, (px, py), 2, hsv_to_bgr(35 + p * 3, 190, 230), -1, cv2.LINE_AA)

    # Núcleo brillante
    core = img.copy()
    cv2.circle(core, (cx, cy), 55, hsv_to_bgr(50, 220, 255), -1, cv2.LINE_AA)
    core = cv2.GaussianBlur(core, (0, 0), 12)
    img[:] = cv2.addWeighted(img, 0.75, core, 0.25, 0)

    cv2.circle(img, (cx, cy), 18, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.putText(
        img,
        "Portal de energia",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (230, 230, 230),
        2,
        cv2.LINE_AA
    )

# Escena 2 - Tormenta geométrica
def scene_nebulosa(img, t):
    background_hsv_gradient(img, t, hue0=5, hue1=35)
    draw_stars(img, seed=44, n=170)
    draw_light_nebula(img, t, seed=45, amount=5)

    cx, cy = W // 2, H // 2

    # Ondas horizontales de fondo
    for y in range(0, H, 18):
        offset = int(30 * math.sin(t * 1.3 + y * 0.04))
        color = hsv_to_bgr(8 + y // 20, 160, 120)

        cv2.line(
            img,
            (0, y + offset),
            (W, y - offset),
            color,
            1,
            cv2.LINE_AA
        )

    # Curva mariposa
    def bx(th):
        return np.sin(th) * (
            np.exp(np.cos(th))
            - 2 * np.cos(4 * th)
            - np.sin(th / 12) ** 5
        )

    def by(th):
        return np.cos(th) * (
            np.exp(np.cos(th))
            - 2 * np.cos(4 * th)
            - np.sin(th / 12) ** 5
        )

    pts = poly_param(
        bx,
        by,
        0,
        12 * math.pi,
        2200,
        cx,
        cy,
        75 + 8 * math.sin(t),
        75 + 8 * math.cos(t)
    )

    # Rotación de la curva con matriz afín
    angle = t * 18
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    pts_float = pts.reshape(-1, 2).astype(np.float32)
    ones = np.ones((pts_float.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts_float, ones])
    pts_rot = (M @ pts_h.T).T.astype(np.int32).reshape((-1, 1, 2))

    cv2.polylines(
        img,
        [pts_rot],
        False,
        hsv_to_bgr(int(10 + 20 * math.sin(t * 0.8)), 230, 245),
        2,
        cv2.LINE_AA
    )

    # Polígonos giratorios
    for i in range(5):
        sides = 3 + i
        radius = 60 + i * 35
        ang0 = t * (0.4 + i * 0.1)

        poly = []

        for j in range(sides):
            ang = 2 * math.pi * j / sides + ang0
            x = int(cx + radius * math.cos(ang))
            y = int(cy + radius * math.sin(ang))
            poly.append([x, y])

        poly = np.array(poly, np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            img,
            [poly],
            True,
            hsv_to_bgr(20 + i * 10, 190, 210),
            1,
            cv2.LINE_AA
        )

    # Rayos suaves de energía
    for i in range(9):
        a = t * 0.8 + i * 0.7
        x1 = int(cx + 65 * math.cos(a))
        y1 = int(cy + 65 * math.sin(a))
        x2 = int(cx + 255 * math.cos(a + 0.25 * math.sin(t + i)))
        y2 = int(cy + 255 * math.sin(a + 0.25 * math.sin(t + i)))
        cv2.line(img, (x1, y1), (x2, y2), hsv_to_bgr(12 + i * 4, 160, 180), 1, cv2.LINE_AA)

    # Halo rojo/naranja
    halo = np.zeros_like(img)

    for r in range(40, 260, 35):
        col = hsv_to_bgr(12 + r // 18, 200, 180)
        cv2.circle(halo, (cx, cy), r, col, 2, cv2.LINE_AA)

    halo = cv2.GaussianBlur(halo, (0, 0), 8)
    img[:] = cv2.addWeighted(img, 0.85, halo, 0.35, 0)

    cv2.putText(
        img,
        "Tormenta geometrica",
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (230, 230, 230),
        2,
        cv2.LINE_AA
    )

# Escena 3 - Transformaciones
def scene_transformaciones(img, t):
    background_hsv_gradient(img, t, hue0=140, hue1=110)
    draw_stars(img, seed=55, n=330)
    draw_twinkle_stars(img, t, seed=56, n=40)

    # Grid futurista sencillo
    for x in range(0, W, 40):
        cv2.line(img, (x, H // 2 + 70), (int(W//2 + (x-W//2)*1.5), H), hsv_to_bgr(120, 90, 90), 1, cv2.LINE_AA)
    for y in range(H // 2 + 80, H, 35):
        cv2.line(img, (0, y), (W, y), hsv_to_bgr(120, 90, 80), 1, cv2.LINE_AA)

    nave = np.array([
        [0, -60],
        [15, -20],
        [15, 30],
        [30, 50],
        [0, 40],
        [-30, 50],
        [-15, 30],
        [-15, -20]
    ], dtype=np.float32)

    ones = np.ones((len(nave), 1), np.float32)
    coords = np.hstack([nave, ones])

    # Rotación + escala
    angle = t * 45
    scale = 1.0 + 0.4 * math.sin(t * 1.2)

    M_rot = cv2.getRotationMatrix2D((0, 0), angle, scale)
    rot = (M_rot @ coords.T).T

    rot[:, 0] += W * 0.25
    rot[:, 1] += H * 0.5

    cv2.fillPoly(img, [rot.astype(np.int32)], hsv_to_bgr(100, 200, 230))
    cv2.polylines(img, [rot.astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(
        img,
        "Rotacion + Escala",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    # Shear
    sh = 0.5 * math.sin(t * 0.8)
    M_shear = np.float32([[1, sh, 0], [0, 1, 0]])
    sheared = (M_shear @ coords.T).T

    sheared[:, 0] += W * 0.75
    sheared[:, 1] += H * 0.5

    cv2.fillPoly(img, [sheared.astype(np.int32)], hsv_to_bgr(20, 200, 230))
    cv2.polylines(img, [sheared.astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(
        img,
        "Shear",
        (int(W * 0.68), 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    # Espejo + composición
    tmp = np.zeros_like(img)

    M_centro = cv2.getRotationMatrix2D((0, 0), t * 30, 0.5)
    nave_r = (M_centro @ coords.T).T

    nave_r[:, 0] += W // 2
    nave_r[:, 1] += H // 4

    cv2.fillPoly(tmp, [nave_r.astype(np.int32)], hsv_to_bgr(60, 200, 220))

    flipped = cv2.flip(tmp, 1)
    img[:] = cv2.addWeighted(img, 1.0, flipped, 0.5, 0)

    cv2.putText(
        img,
        "Espejo",
        (int(W * 0.44), int(H * 0.12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    cv2.line(img, (W // 2, 60), (W // 2, H - 60), (150, 150, 150), 1, cv2.LINE_AA)

# Escena 4 - Campo de asteroides
def scene_asteroides(img, t):
    background_hsv_gradient(img, t, hue0=150, hue1=120)
    draw_stars(img, seed=77, n=200)

    # Semilla basada en el frame para que sea reproducible
    rng = np.random.default_rng(int(t * 30) + 100)

    n = 650
    xs = rng.random(n) * W
    ys = rng.random(n) * H

    dx = xs - W * 0.5
    dy = ys - H * 0.5
    dist = np.sqrt(dx ** 2 + dy ** 2) + 1

    vx = (-dy / dist) * 1.8 + np.sin(ys / 60.0 + t * 1.2) * 1.2
    vy = (dx / dist) * 1.8 + np.cos(xs / 60.0 + t * 0.9) * 1.2

    xs = (xs + vx * 45) % W
    ys = (ys + vy * 45) % H

    hue_base = int(145 + 25 * math.sin(t * 0.5))

    for i in range(0, n, 2):
        speed = math.sqrt(float(vx[i]) ** 2 + float(vy[i]) ** 2)
        col = hsv_to_bgr(hue_base + int(speed * 8), 180, 210)
        cv2.circle(img, (int(xs[i]), int(ys[i])), 1, col, -1)
        if i % 34 == 0:
            x0, y0 = int(xs[i]), int(ys[i])
            x1, y1 = int(xs[i] - vx[i] * 3), int(ys[i] - vy[i] * 3)
            cv2.line(img, (x0, y0), (x1, y1), col, 1, cv2.LINE_AA)

    img[:] = cv2.GaussianBlur(img, (0, 0), 1.0)
    draw_planet(img, W // 2, H // 2, 35, hue=110, rings=True)

# Escena 5 - Final
def scene_final(img, t):
    background_hsv_gradient(img, t, hue0=155, hue1=125)
    draw_stars(img, seed=99, n=430)
    draw_light_nebula(img, t, seed=100, amount=5)
    draw_twinkle_stars(img, t, seed=101, n=55)

    draw_planet(img, W // 2, H // 2, 45, hue=15)

    glow = img.copy()
    cv2.circle(glow, (W // 2, H // 2), 70, hsv_to_bgr(20, 220, 255), -1, cv2.LINE_AA)
    img[:] = cv2.addWeighted(img, 0.8, cv2.GaussianBlur(glow, (0, 0), 18), 0.2, 0)

    # Lemniscata
    a = 2.8 + 0.3 * math.sin(t * 0.6)

    def lx(th):
        den = 1 + np.sin(th) ** 2
        return a * np.cos(th) / den

    def ly(th):
        den = 1 + np.sin(th) ** 2
        return a * np.sin(th) * np.cos(th) / den

    pts = poly_param(lx, ly, 0, 2 * math.pi, 1000, W // 2, H // 2, 200, 200)

    cv2.polylines(
        img,
        [pts],
        True,
        hsv_to_bgr(int(150 + 15 * math.sin(t * 0.5)), 180, 200),
        1,
        cv2.LINE_AA
    )

    # Hipotrocoide
    R, r, d = 7.0, 2.0, 5.0
    w = (R - r) / r
    phase = t * 0.3

    def hx(th):
        return (R - r) * np.cos(th) + d * np.cos(w * th + phase)

    def hy(th):
        return (R - r) * np.sin(th) - d * np.sin(w * th + phase)

    pts2 = poly_param(hx, hy, 0, 10 * math.pi, 2000, W // 2, H // 2, 30, 30)

    cv2.polylines(
        img,
        [pts2],
        False,
        hsv_to_bgr(int(10 + 140 * (0.5 + 0.5 * math.sin(t * 0.4))), 230, 240),
        2,
        cv2.LINE_AA
    )

    datos_planetas = [
        (130, 0.9, 8, 100),
        (200, 0.5, 14, 20),
        (280, 0.3, 10, 50)
    ]

    for dist, vel, rad, hue in datos_planetas:
        px = int(W // 2 + dist * math.cos(t * vel))
        py = int(H // 2 + dist // 3 * math.sin(t * vel))
        draw_planet(img, px, py, rad, hue)

    # Máscara circular para enfocar el centro
    mask = np.zeros((H, W), np.uint8)
    r_m = int(270 + 20 * math.sin(t * 0.8))
    cv2.circle(mask, (W // 2, H // 2), r_m, 255, -1)

    mask3 = cv2.merge([mask, mask, mask])
    blurred = cv2.GaussianBlur(img, (0, 0), 10)

    img[:] = np.where(
        mask3 > 0,
        img,
        (blurred.astype(np.float32) * 0.3).astype(np.uint8)
    )

    fade = smoothstep(1.5, 4.0, t)

    if fade > 0.01:
        overlay = img.copy()

        cv2.putText(
            overlay,
            "FIN",
            (340, H // 2 + 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        draw_text_glow(overlay, "FIN", (340, H // 2 + 180), 2.0, (255, 255, 255), 3, hsv_to_bgr(20, 100, 255))
        img[:] = cv2.addWeighted(img, 1 - fade, overlay, fade, 0)

# Render de escenas
def render_scene(buf, scene_id, t):
    if scene_id == 0:
        scene_credits(buf, t)
    elif scene_id == 1:
        scene_orbitas(buf, t)
    elif scene_id == 2:
        scene_nebulosa(buf, t)
    elif scene_id == 3:
        scene_transformaciones(buf, t)
    elif scene_id == 4:
        scene_asteroides(buf, t)
    else:
        scene_final(buf, t)

# Timeline principal
def timeline(t, bufA, bufB):
    block = int(min(5, max(0, t // 10)))
    t_in = t - block * 10

    render_scene(bufA, block, t_in)
    frame = bufA

    # Transición entre escenas
    if block < 5 and t_in >= 8.8:
        render_scene(bufA, block, t_in)
        render_scene(bufB, block + 1, t_in)

        a = smoothstep(8.8, 10.0, t_in)
        frame = cv2.addWeighted(bufA, 1 - a, bufB, a, 0)

        flash = smoothstep(9.6, 10.0, t_in)

        if flash > 0:
            frame = cv2.addWeighted(
                frame,
                1.0,
                np.full_like(frame, 255),
                0.10 * flash,
                0
            )

    # Fade global
    fin = smoothstep(0.0, 1.5, t)
    fout = 1.0 - smoothstep(DURATION - 1.5, DURATION, t)
    f = fin * fout

    if f < 0.999:
        frame = (frame.astype(np.float32) * f).astype(np.uint8)

    return frame

# Programa principal
def main():
    bufA = np.zeros((H, W, 3), np.uint8)
    bufB = np.zeros((H, W, 3), np.uint8)

    guardar_mascaras()

    total_frames = int(DURATION * FPS)

    video_path = os.path.join(RENDERS_DIR, "demo_cosmos_ligero.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (W, H))

    print("Guardando archivos en:", RENDERS_DIR)

    t0 = time.perf_counter()

    for i in range(total_frames):
        t = i / FPS

        frame = timeline(t, bufA, bufB)

        # Postprocesamiento
        frame = post_vignette(frame, 0.62)
        frame = post_scanlines(frame, 0.07)
        frame = post_posterize(frame, 22)

        writer.write(frame)

        # Capturas automáticas al segundo 5 de cada escena
        for sid in range(6):
            capture_time = sid * 10 + 5.0

            if abs(t - capture_time) < 1 / FPS:
                cap_path = os.path.join(RENDERS_DIR, f"escena_{sid:02d}.png")
                cv2.imwrite(cap_path, frame)
                print("Captura guardada:", f"escena_{sid:02d}.png")

        if SHOW_WINDOW:
            cv2.imshow("Cosmos Procedural - OpenCV", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    writer.release()

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("Listo.")
    print("Video generado en:", video_path)
    print("Tiempo de render:", round(time.perf_counter() - t0, 2), "segundos")

if __name__ == "__main__":
    main()