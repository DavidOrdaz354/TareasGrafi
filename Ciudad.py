import os
import sys
import math
import glfw
import cv2
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *

# Desactivar logs innecesarios
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

# ── MediaPipe Tasks API ───────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# ── Estado de cámara 3D ───────────────────────────────────────
angle_x, angle_y = 38.0, 45.0
zoom = -70.0
target_zoom = -70.0
pan_x, pan_y = 0.0, 0.0

prev_right_index = None
prev_left_index = None

MIN_ZOOM = -140.0   # más lejos
MAX_ZOOM = -8.0     # más cerca


# ── Primitivas base ───────────────────────────────────────────
def draw_generic_cube(w, h, d, r, g, b):
    w_h, d_h = w / 2.0, d / 2.0
    glBegin(GL_QUADS)

    # Frente
    glColor3f(r, g, b)
    glVertex3f(-w_h, 0, d_h)
    glVertex3f(w_h, 0, d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(-w_h, h, d_h)

    # Atrás
    glColor3f(r * 0.82, g * 0.82, b * 0.82)
    glVertex3f(-w_h, 0, -d_h)
    glVertex3f(w_h, 0, -d_h)
    glVertex3f(w_h, h, -d_h)
    glVertex3f(-w_h, h, -d_h)

    # Izquierda
    glColor3f(r * 0.72, g * 0.72, b * 0.72)
    glVertex3f(-w_h, 0, -d_h)
    glVertex3f(-w_h, 0, d_h)
    glVertex3f(-w_h, h, d_h)
    glVertex3f(-w_h, h, -d_h)

    # Derecha
    glColor3f(r * 0.72, g * 0.72, b * 0.72)
    glVertex3f(w_h, 0, -d_h)
    glVertex3f(w_h, 0, d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(w_h, h, -d_h)

    # Arriba
    glColor3f(r * 0.92, g * 0.92, b * 0.92)
    glVertex3f(-w_h, h, -d_h)
    glVertex3f(w_h, h, -d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(-w_h, h, d_h)

    glEnd()


def draw_pyramid(w, h, d, r, g, b):
    w_h, d_h = w / 2.0, d / 2.0
    glBegin(GL_TRIANGLES)

    glColor3f(r, g, b)
    glVertex3f(-w_h, 0, d_h)
    glVertex3f(w_h, 0, d_h)
    glVertex3f(0, h, 0)

    glColor3f(r * 0.82, g * 0.82, b * 0.82)
    glVertex3f(-w_h, 0, -d_h)
    glVertex3f(w_h, 0, -d_h)
    glVertex3f(0, h, 0)

    glColor3f(r * 0.72, g * 0.72, b * 0.72)
    glVertex3f(-w_h, 0, -d_h)
    glVertex3f(-w_h, 0, d_h)
    glVertex3f(0, h, 0)

    glVertex3f(w_h, 0, -d_h)
    glVertex3f(w_h, 0, d_h)
    glVertex3f(0, h, 0)

    glEnd()


def draw_windows(w, h, d, rows, cols):
    w_half, d_half = w / 2.0, d / 2.0
    win_w = w / (cols * 2)
    win_h = h / (rows * 2)

    glColor3f(0.97, 0.95, 0.45)
    glBegin(GL_QUADS)
    z_f = d_half + 0.01

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 3 == 0:
                continue
            x = -w_half + (c * (w / cols)) + win_w / 2
            y = (r * (h / rows)) + win_h / 2
            glVertex3f(x, y, z_f)
            glVertex3f(x + win_w, y, z_f)
            glVertex3f(x + win_w, y + win_h, z_f)
            glVertex3f(x, y + win_h, z_f)

    glEnd()


# ── Casas / edificios ─────────────────────────────────────────
def draw_detailed_house(w, h, d, r, g, b):
    draw_generic_cube(w, h, d, r, g, b)

    # puerta
    glPushMatrix()
    glTranslatef(0, 0, d / 2 + 0.02)
    draw_generic_cube(w * 0.22, h * 0.55, 0.05, 0.35, 0.2, 0.1)
    glPopMatrix()

    # ventanas
    glPushMatrix()
    glTranslatef(0, 0.35, 0)
    draw_windows(w * 0.8, h * 0.55, d, 2, 2)
    glPopMatrix()

    # techo
    w_h, d_h = (w + 0.3) / 2.0, (d + 0.3) / 2.0
    roof_y = h + 0.9
    glBegin(GL_TRIANGLES)
    glColor3f(0.82, 0.28, 0.12)
    glVertex3f(-w_h, h, d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(0, roof_y, 0)

    glVertex3f(-w_h, h, -d_h)
    glVertex3f(w_h, h, -d_h)
    glVertex3f(0, roof_y, 0)
    glEnd()


def draw_apartment_block(w, h, d, r, g, b):
    draw_generic_cube(w, h, d, r, g, b)
    draw_windows(w, h * 0.95, d, max(4, int(h // 1.6)), max(3, int(w // 1.3)))


def draw_school():
    draw_generic_cube(12.0, 5.0, 5.0, 0.72, 0.72, 0.74)
    glPushMatrix()
    glTranslatef(4.0, 0, 4.0)
    draw_generic_cube(4.0, 5.0, 4.0, 0.72, 0.72, 0.74)
    glPopMatrix()
    draw_windows(10.0, 4.0, 5.0, 3, 5)

    glPushMatrix()
    glTranslatef(0, 5.0, 2.0)
    draw_generic_cube(2.5, 1.0, 0.5, 0.6, 0.1, 0.1)
    glPopMatrix()


def draw_church():
    draw_generic_cube(6.0, 6.0, 10.0, 0.85, 0.82, 0.75)

    glPushMatrix()
    glTranslatef(0, 0, 4.0)
    draw_generic_cube(3.0, 11.0, 3.0, 0.75, 0.72, 0.65)
    glTranslatef(0, 11.0, 0)
    draw_pyramid(3.4, 3.0, 3.4, 0.3, 0.3, 0.35)
    glTranslatef(0, 3.0, 0)
    draw_generic_cube(0.2, 1.2, 0.2, 0.9, 0.8, 0.2)
    glTranslatef(0, 0.4, 0)
    draw_generic_cube(0.8, 0.2, 0.2, 0.9, 0.8, 0.2)
    glPopMatrix()


def draw_small_playground():
    glBegin(GL_QUADS)
    glColor3f(0.35, 0.65, 0.3)
    glVertex3f(-7.0, 0.02, -7.0)
    glVertex3f(7.0, 0.02, -7.0)
    glVertex3f(7.0, 0.02, 7.0)
    glVertex3f(-7.0, 0.02, 7.0)
    glEnd()

    # columpio
    glPushMatrix()
    glTranslatef(-3.5, 0, -2.0)
    draw_generic_cube(0.1, 2.0, 0.1, 0.2, 0.2, 0.2)
    glTranslatef(4.0, 0, 0)
    draw_generic_cube(0.1, 2.0, 0.1, 0.2, 0.2, 0.2)
    glTranslatef(-2.0, 2.0, 0)
    draw_generic_cube(4.2, 0.1, 0.1, 0.2, 0.2, 0.2)
    glTranslatef(0, -1.3, 0)
    draw_generic_cube(1.0, 0.08, 0.4, 0.8, 0.2, 0.2)
    glPopMatrix()

    # resbaladilla
    glPushMatrix()
    glTranslatef(2.5, 0, -2.0)
    draw_generic_cube(0.8, 1.6, 0.8, 0.2, 0.4, 0.8)
    glPushMatrix()
    glTranslatef(0, 0.7, 1.2)
    glRotatef(30, 1, 0, 0)
    draw_generic_cube(0.7, 0.1, 2.2, 0.85, 0.85, 0.85)
    glPopMatrix()
    glPopMatrix()

    # pasamanos
    glPushMatrix()
    glTranslatef(0, 0, 3.0)
    draw_generic_cube(0.1, 1.8, 0.1, 0.8, 0.6, 0.1)
    glTranslatef(0, 0, -4.0)
    draw_generic_cube(0.1, 1.8, 0.1, 0.8, 0.6, 0.1)
    glTranslatef(0, 1.8, 2.0)
    draw_generic_cube(0.12, 0.1, 4.2, 0.8, 0.6, 0.1)
    glPopMatrix()


def draw_kiosk():
    glPushMatrix()
    draw_generic_cube(3.2, 0.6, 3.2, 0.5, 0.35, 0.25)

    for px, pz in [(-1.4, -1.4), (1.4, -1.4), (1.4, 1.4), (-1.4, 1.4)]:
        glPushMatrix()
        glTranslatef(px, 0.6, pz)
        draw_generic_cube(0.15, 2.2, 0.15, 0.82, 0.73, 0.55)
        glPopMatrix()

    glTranslatef(0, 2.8, 0)
    draw_pyramid(3.6, 1.5, 3.6, 0.7, 0.2, 0.2)
    glPopMatrix()


# ── Infraestructura ───────────────────────────────────────────
def draw_street_lamp():
    glPushMatrix()
    draw_generic_cube(0.2, 4.0, 0.2, 0.2, 0.2, 0.22)
    glTranslatef(0, 4.0, 0.35)
    draw_generic_cube(0.2, 0.15, 0.9, 0.2, 0.2, 0.22)
    glTranslatef(0, -0.05, 0.35)
    draw_generic_cube(0.45, 0.22, 0.45, 0.95, 0.95, 0.5)
    glPopMatrix()


def draw_traffic_light():
    glPushMatrix()
    draw_generic_cube(0.15, 3.5, 0.15, 0.15, 0.15, 0.15)
    glTranslatef(0, 3.5, 0)
    draw_generic_cube(0.4, 1.0, 0.4, 0.1, 0.1, 0.1)

    glTranslatef(0, 0.3, 0.21)
    draw_generic_cube(0.2, 0.18, 0.02, 0.9, 0.1, 0.1)
    glTranslatef(0, -0.25, 0)
    draw_generic_cube(0.2, 0.18, 0.02, 0.9, 0.8, 0.1)
    glTranslatef(0, -0.25, 0)
    draw_generic_cube(0.2, 0.18, 0.02, 0.1, 0.8, 0.1)
    glPopMatrix()


def draw_bench():
    glPushMatrix()
    draw_generic_cube(1.2, 0.12, 0.35, 0.45, 0.28, 0.1)
    glPushMatrix()
    glTranslatef(-0.45, -0.01, 0)
    draw_generic_cube(0.08, 0.45, 0.08, 0.2, 0.2, 0.2)
    glTranslatef(0.9, 0, 0)
    draw_generic_cube(0.08, 0.45, 0.08, 0.2, 0.2, 0.2)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0, 0.32, -0.12)
    draw_generic_cube(1.2, 0.12, 0.15, 0.45, 0.28, 0.1)
    glPopMatrix()
    glPopMatrix()


def draw_crosswalk(width=4.6, depth=3.0, stripes=6, horizontal=False):
    stripe_w = width / (stripes * 2)
    glColor3f(0.95, 0.95, 0.95)
    glBegin(GL_QUADS)
    for i in range(stripes):
        if not horizontal:
            x0 = -width / 2 + i * stripe_w * 2
            x1 = x0 + stripe_w
            glVertex3f(x0, 0.03, -depth / 2)
            glVertex3f(x1, 0.03, -depth / 2)
            glVertex3f(x1, 0.03, depth / 2)
            glVertex3f(x0, 0.03, depth / 2)
        else:
            z0 = -width / 2 + i * stripe_w * 2
            z1 = z0 + stripe_w
            glVertex3f(-depth / 2, 0.03, z0)
            glVertex3f(depth / 2, 0.03, z0)
            glVertex3f(depth / 2, 0.03, z1)
            glVertex3f(-depth / 2, 0.03, z1)
    glEnd()


# ── Personajes / mascotas ─────────────────────────────────────
def draw_person(r, g, b):
    glPushMatrix()

    # piernas
    glPushMatrix()
    glTranslatef(-0.06, 0, 0)
    draw_generic_cube(0.06, 0.38, 0.06, 0.12, 0.12, 0.15)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.06, 0, 0)
    draw_generic_cube(0.06, 0.38, 0.06, 0.12, 0.12, 0.15)
    glPopMatrix()

    # torso
    glPushMatrix()
    glTranslatef(0, 0.38, 0)
    draw_generic_cube(0.24, 0.34, 0.12, r, g, b)
    glPopMatrix()

    # brazos
    glPushMatrix()
    glTranslatef(-0.16, 0.42, 0)
    draw_generic_cube(0.05, 0.28, 0.05, 0.86, 0.73, 0.62)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.16, 0.42, 0)
    draw_generic_cube(0.05, 0.28, 0.05, 0.86, 0.73, 0.62)
    glPopMatrix()

    # cabeza
    glPushMatrix()
    glTranslatef(0, 0.72, 0)
    draw_generic_cube(0.18, 0.2, 0.18, 0.88, 0.75, 0.64)
    glPopMatrix()

    glPopMatrix()


def draw_dog():
    glPushMatrix()
    draw_generic_cube(0.3, 0.3, 0.6, 0.55, 0.27, 0.07)

    glPushMatrix()
    glTranslatef(-0.1, -0.15, 0.2)
    draw_generic_cube(0.08, 0.2, 0.08, 0.4, 0.2, 0.0)
    glTranslatef(0.2, 0, 0)
    draw_generic_cube(0.08, 0.2, 0.08, 0.4, 0.2, 0.0)
    glTranslatef(0, 0, -0.4)
    draw_generic_cube(0.08, 0.2, 0.08, 0.4, 0.2, 0.0)
    glTranslatef(-0.2, 0, 0)
    draw_generic_cube(0.08, 0.2, 0.08, 0.4, 0.2, 0.0)
    glPopMatrix()

    glTranslatef(0, 0.25, 0.25)
    draw_generic_cube(0.25, 0.25, 0.25, 0.55, 0.27, 0.07)
    glTranslatef(0, 0.05, -0.1)
    draw_generic_cube(0.32, 0.1, 0.1, 0.3, 0.15, 0.0)
    glPopMatrix()


# ── Vehículos ─────────────────────────────────────────────────
def draw_truck():
    glPushMatrix()
    draw_generic_cube(1.6, 1.8, 4.0, 0.85, 0.85, 0.85)
    glTranslatef(0, 0, 2.3)
    draw_generic_cube(1.5, 1.1, 1.2, 0.8, 0.1, 0.1)
    glPopMatrix()


def draw_motorcycle(r, g, b):
    glPushMatrix()
    draw_generic_cube(0.4, 0.45, 1.2, r, g, b)
    glTranslatef(0, -0.14, 0.45)
    draw_generic_cube(0.22, 0.26, 0.28, 0.05, 0.05, 0.05)
    glTranslatef(0, 0, -0.9)
    draw_generic_cube(0.22, 0.26, 0.28, 0.05, 0.05, 0.05)
    glPopMatrix()


def draw_car(r, g, b):
    glPushMatrix()
    draw_generic_cube(1.15, 0.42, 2.05, r, g, b)
    glPushMatrix()
    glTranslatef(0, 0.42, -0.1)
    draw_generic_cube(0.9, 0.35, 1.1, r * 0.6, g * 0.6, b * 0.6)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0, 0, -1.1)
    draw_generic_cube(1.25, 0.24, 0.38, 0.05, 0.05, 0.05)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0, 0, 1.1)
    draw_generic_cube(1.25, 0.24, 0.38, 0.05, 0.05, 0.05)
    glPopMatrix()

    glPopMatrix()


# ── Canchas / vegetación ──────────────────────────────────────
def draw_volleyball_court():
    glBegin(GL_QUADS)
    glColor3f(0.9, 0.5, 0.2)
    glVertex3f(-4.0, 0.02, -7.0)
    glVertex3f(4.0, 0.02, -7.0)
    glVertex3f(4.0, 0.02, 7.0)
    glVertex3f(-4.0, 0.02, 7.0)

    glColor3f(0.1, 0.4, 0.7)
    glVertex3f(-3.0, 0.025, -6.0)
    glVertex3f(3.0, 0.025, -6.0)
    glVertex3f(3.0, 0.025, 6.0)
    glVertex3f(-3.0, 0.025, 6.0)
    glEnd()

    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 1.0)
    glVertex3f(-3.0, 0.03, 0.0)
    glVertex3f(3.0, 0.03, 0.0)
    glEnd()

    draw_generic_cube(0.1, 2.2, 0.1, 0.6, 0.6, 0.6)
    glPushMatrix()
    glTranslatef(0, 1.3, 0)
    draw_generic_cube(6.0, 0.7, 0.02, 0.9, 0.9, 0.9)
    glPopMatrix()


def draw_basketball_court():
    glBegin(GL_QUADS)
    glColor3f(0.75, 0.52, 0.3)
    glVertex3f(-4.0, 0.02, -8.0)
    glVertex3f(4.0, 0.02, -8.0)
    glVertex3f(4.0, 0.02, 8.0)
    glVertex3f(-4.0, 0.02, 8.0)
    glEnd()

    glPushMatrix()
    glTranslatef(0, 0, -7.5)
    draw_generic_cube(0.15, 3.0, 0.15, 0.2, 0.2, 0.2)
    glTranslatef(0, 3.0, 0.2)
    draw_generic_cube(1.8, 1.1, 0.05, 1.0, 1.0, 1.0)
    glTranslatef(0, -0.3, 0.2)
    draw_generic_cube(0.5, 0.1, 0.5, 0.9, 0.1, 0.1)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0, 0, 7.5)
    draw_generic_cube(0.15, 3.0, 0.15, 0.2, 0.2, 0.2)
    glTranslatef(0, 3.0, -0.2)
    draw_generic_cube(1.8, 1.1, 0.05, 1.0, 1.0, 1.0)
    glTranslatef(0, -0.3, -0.2)
    draw_generic_cube(0.5, 0.1, 0.5, 0.9, 0.1, 0.1)
    glPopMatrix()


def draw_tree_round():
    draw_generic_cube(0.25, 1.0, 0.25, 0.4, 0.25, 0.15)
    glPushMatrix()
    glTranslatef(0, 1.0, 0)
    draw_generic_cube(1.2, 1.1, 1.2, 0.2, 0.55, 0.2)
    glPopMatrix()


def draw_tree_square():
    draw_generic_cube(0.25, 0.8, 0.25, 0.35, 0.2, 0.1)
    glPushMatrix()
    glTranslatef(0, 0.8, 0)
    draw_generic_cube(0.9, 1.4, 0.9, 0.15, 0.45, 0.15)
    glPopMatrix()


def draw_tree_pine():
    draw_generic_cube(0.25, 0.7, 0.25, 0.4, 0.25, 0.15)
    glPushMatrix()
    glTranslatef(0, 0.6, 0)
    draw_pyramid(1.5, 1.0, 1.5, 0.1, 0.38, 0.15)
    glTranslatef(0, 0.6, 0)
    draw_pyramid(1.1, 0.9, 1.1, 0.12, 0.42, 0.18)
    glTranslatef(0, 0.5, 0)
    draw_pyramid(0.7, 0.7, 0.7, 0.15, 0.48, 0.22)
    glPopMatrix()


def draw_tree_double_sphere():
    draw_generic_cube(0.2, 1.8, 0.2, 0.4, 0.25, 0.15)
    glPushMatrix()
    glTranslatef(0, 0.8, 0)
    draw_generic_cube(0.9, 0.7, 0.9, 0.25, 0.6, 0.25)
    glTranslatef(0, 0.7, 0)
    draw_generic_cube(0.7, 0.5, 0.7, 0.3, 0.65, 0.3)
    glPopMatrix()


# ── Dibujado auxiliar de calles ───────────────────────────────
def draw_lane_markers_vertical(x, z0, z1):
    glColor3f(0.95, 0.85, 0.2)
    glBegin(GL_QUADS)
    z = z0
    while z < z1:
        glVertex3f(x - 0.08, 0.025, z)
        glVertex3f(x + 0.08, 0.025, z)
        glVertex3f(x + 0.08, 0.025, z + 2.0)
        glVertex3f(x - 0.08, 0.025, z + 2.0)
        z += 4.0
    glEnd()


def draw_lane_markers_horizontal(z, x0, x1):
    glColor3f(0.95, 0.85, 0.2)
    glBegin(GL_QUADS)
    x = x0
    while x < x1:
        glVertex3f(x, 0.025, z - 0.08)
        glVertex3f(x + 2.0, 0.025, z - 0.08)
        glVertex3f(x + 2.0, 0.025, z + 0.08)
        glVertex3f(x, 0.025, z + 0.08)
        x += 4.0
    glEnd()


# ── Escenario principal ───────────────────────────────────────
def draw_scenery():
    # Terreno
    glBegin(GL_QUADS)
    glColor3f(0.14, 0.14, 0.15)
    glVertex3f(-85, -0.01, 85)
    glVertex3f(85, -0.01, 85)
    glVertex3f(85, -0.01, -85)
    glVertex3f(-85, -0.01, -85)
    glEnd()

    # Parque principal grande
    glBegin(GL_QUADS)
    glColor3f(0.18, 0.42, 0.22)
    glVertex3f(-78, 0.01, 72)
    glVertex3f(-22, 0.01, 72)
    glVertex3f(-22, 0.01, -72)
    glVertex3f(-78, 0.01, -72)
    glEnd()

    # Árboles del parque
    for z in range(-66, 67, 7):
        glPushMatrix(); glTranslatef(-72, 0, z); draw_tree_round(); glPopMatrix()
        glPushMatrix(); glTranslatef(-64, 0, z); draw_tree_square(); glPopMatrix()

        if z < -12 or z > 12:
            glPushMatrix(); glTranslatef(-54, 0, z); draw_tree_pine(); glPopMatrix()
            glPushMatrix(); glTranslatef(-44, 0, z); draw_tree_double_sphere(); glPopMatrix()

    # Canchas
    glPushMatrix(); glTranslatef(-47, 0, -12); draw_volleyball_court(); glPopMatrix()
    glPushMatrix(); glTranslatef(-47, 0, 12); draw_basketball_court(); glPopMatrix()

    # Kiosko y bancas
    glPushMatrix(); glTranslatef(-58, 0, -32); draw_kiosk(); glPopMatrix()
    glPushMatrix(); glTranslatef(-64, 0, -27); draw_bench(); glPopMatrix()
    glPushMatrix(); glTranslatef(-51, 0, -27); draw_bench(); glRotatef(180, 0, 1, 0); draw_bench(); glPopMatrix()

    # Personas en parque
    people_park = [
        (-60, -20), (-57, -18), (-55, -35), (-62, -34), (-46, -5),
        (-41, 5), (-48, 22), (-53, 24), (-61, 30), (-58, 8)
    ]
    colors_people = [
        (0.8, 0.2, 0.2), (0.2, 0.35, 0.85), (0.15, 0.7, 0.2),
        (0.85, 0.6, 0.15), (0.6, 0.2, 0.75)
    ]
    for i, (px, pz) in enumerate(people_park):
        glPushMatrix()
        glTranslatef(px, 0, pz)
        draw_person(*colors_people[i % len(colors_people)])
        glPopMatrix()

    # Escuela
    glPushMatrix()
    glTranslatef(54.0, 0, 56.0)
    draw_school()
    glPopMatrix()

    # Iglesia
    glPushMatrix()
    glTranslatef(67.0, 0, 18.0)
    draw_church()
    glPopMatrix()

    # Parque infantil
    glPushMatrix()
    glTranslatef(30.0, 0, -8.0)
    draw_small_playground()
    glTranslatef(-2.0, 0.3, 1.0)
    draw_dog()
    glPopMatrix()

    # Gente en el parque infantil
    for px, pz, cr, cg, cb in [
        (24, -6, 0.2, 0.4, 0.8),
        (28, 2, 0.8, 0.3, 0.2),
        (34, -4, 0.2, 0.75, 0.25),
        (36, 4, 0.7, 0.6, 0.15),
    ]:
        glPushMatrix()
        glTranslatef(px, 0, pz)
        draw_person(cr, cg, cb)
        glPopMatrix()

    # Calles principales
    glColor3f(0.28, 0.28, 0.30)
    glBegin(GL_QUADS)

    roads_x = [-6, 10, 26, 42, 58, 74]
    roads_z = [-58, -42, -26, -10, 6, 22, 38, 54, 70]

    for x in roads_x:
        glVertex3f(x - 1.6, 0.015, 74)
        glVertex3f(x + 1.6, 0.015, 74)
        glVertex3f(x + 1.6, 0.015, -74)
        glVertex3f(x - 1.6, 0.015, -74)

    for z in roads_z:
        glVertex3f(-20, 0.015, z + 1.6)
        glVertex3f(78, 0.015, z + 1.6)
        glVertex3f(78, 0.015, z - 1.6)
        glVertex3f(-20, 0.015, z - 1.6)

    glEnd()

    # Rayas centrales
    for x in roads_x:
        draw_lane_markers_vertical(x, -72, 72)
    for z in roads_z:
        draw_lane_markers_horizontal(z, -18, 76)

    # Cruces peatonales
    for x, z in [(10, -10), (26, 22), (42, -26), (58, 38), (74, 6)]:
        glPushMatrix(); glTranslatef(x, 0, z); draw_crosswalk(); glPopMatrix()
        glPushMatrix(); glTranslatef(x, 0, z); draw_crosswalk(horizontal=True); glPopMatrix()

    # Postes de luz
    for x in [-12, 4, 20, 36, 52, 68]:
        for z in [-50, -18, 14, 46]:
            glPushMatrix()
            glTranslatef(x, 0, z)
            draw_street_lamp()
            glPopMatrix()

    # Semáforos
    for x, z in [(10, -10), (26, 22), (42, -26), (58, 38), (74, 6)]:
        glPushMatrix(); glTranslatef(x - 2.2, 0, z - 2.2); draw_traffic_light(); glPopMatrix()
        glPushMatrix(); glTranslatef(x + 2.2, 0, z + 2.2); draw_traffic_light(); glPopMatrix()

    # Más tráfico: carros
    cars = [
        (-6, -49, 0.85, 0.1, 0.1),
        (10, -33, 0.1, 0.3, 0.8),
        (26, -17, 0.9, 0.8, 0.1),
        (42, -1, 0.15, 0.6, 0.2),
        (58, 15, 0.95, 0.95, 0.95),
        (74, 31, 0.9, 0.4, 0.0),
        (10, 47, 0.4, 0.1, 0.6),
        (26, 63, 0.1, 0.1, 0.1),
        (42, -49, 0.5, 0.35, 0.05),
        (58, -33, 0.15, 0.75, 0.75),
        (74, -17, 0.85, 0.25, 0.5),
        (-6, 15, 0.2, 0.5, 0.9),
        (26, 31, 0.75, 0.2, 0.2),
        (58, 63, 0.2, 0.7, 0.25),
    ]
    for x, z, r, g, b in cars:
        glPushMatrix()
        glTranslatef(x, 0.1, z)
        draw_car(r, g, b)
        glPopMatrix()

    # Camiones
    for x, z in [(-6, -2), (42, 47), (74, -49)]:
        glPushMatrix()
        glTranslatef(x, 0.1, z)
        draw_truck()
        glPopMatrix()

    # Motos
    motos = [
        (10, -57, 0.1, 0.8, 0.8),
        (26, -41, 0.9, 0.1, 0.5),
        (42, -25, 0.1, 0.9, 0.2),
        (58, -9, 0.9, 0.55, 0.0),
        (74, 7, 0.5, 0.2, 0.9),
        (10, 23, 0.95, 0.2, 0.2),
        (26, 39, 0.2, 0.45, 0.95),
        (42, 55, 0.9, 0.85, 0.1)
    ]
    for x, z, r, g, b in motos:
        glPushMatrix()
        glTranslatef(x, 0.15, z)
        draw_motorcycle(r, g, b)
        glPopMatrix()

    # Distrito comercial / rascacielos
    skyscrapers = [
        (-4, -64, 4.5, 18.0, 4.5, 0.25, 0.35, 0.5),
        (12, -64, 4.0, 24.0, 4.0, 0.2, 0.2, 0.3),
        (28, -64, 5.0, 28.0, 5.0, 0.15, 0.4, 0.45),
        (44, -64, 4.2, 16.0, 4.2, 0.3, 0.3, 0.35),
        (60, -64, 4.4, 20.0, 4.4, 0.25, 0.28, 0.45),
        (-4, -48, 4.0, 15.0, 4.0, 0.35, 0.35, 0.4),
        (12, -48, 5.5, 34.0, 5.5, 0.1, 0.25, 0.5),
        (28, -48, 4.2, 21.0, 4.2, 0.3, 0.2, 0.2),
        (44, -48, 5.0, 26.0, 5.0, 0.2, 0.36, 0.3),
        (60, -48, 4.0, 17.0, 4.0, 0.4, 0.32, 0.32),
        (-4, -32, 3.8, 13.0, 3.8, 0.28, 0.32, 0.36),
        (12, -32, 5.0, 22.0, 5.0, 0.18, 0.24, 0.42),
    ]

    for x, z, w, h, d, r, g, b in skyscrapers:
        glPushMatrix()
        glTranslatef(x, 0, z)
        draw_apartment_block(w, h, d, r, g, b)
        glPopMatrix()

    # Edificios medianos
    for x, z, h in [(60, -16, 10), (44, 0, 9), (28, 16, 11), (12, 32, 8), (-4, 16, 9)]:
        glPushMatrix()
        glTranslatef(x, 0, z)
        draw_apartment_block(4.0, h, 4.0, 0.62, 0.64, 0.68)
        glPopMatrix()

    # Zona residencial más grande y densa
    house_colors = [
        (0.85, 0.45, 0.45), (0.45, 0.65, 0.85), (0.55, 0.75, 0.55),
        (0.85, 0.80, 0.55), (0.75, 0.60, 0.80), (0.80, 0.80, 0.80),
        (0.95, 0.65, 0.45), (0.65, 0.85, 0.78)
    ]

    for hz in range(-2, 69, 7):
        for hx in range(-2, 69, 7):
            # deja libres las calles
            near_vertical = any(abs(hx - rx) < 4 for rx in roads_x)
            near_horizontal = any(abs(hz - rz) < 4 for rz in roads_z)
            if near_vertical or near_horizontal:
                continue

            index_seed = abs(int(hx * 17 + hz * 29))
            w_var = 2.3 + (index_seed % 3) * 0.45
            h_var = 2.0 + ((index_seed + 1) % 3) * 0.45
            d_var = 2.3 + ((index_seed + 2) % 3) * 0.45

            offset_x = ((index_seed % 5) - 2) * 0.18
            offset_z = (((index_seed + 2) % 5) - 2) * 0.18
            r_c, g_c, b_c = house_colors[index_seed % len(house_colors)]

            glPushMatrix()
            glTranslatef(hx + offset_x, 0, hz + offset_z)
            draw_detailed_house(w_var, h_var, d_var, r_c, g_c, b_c)
            glPopMatrix()

            # árbol o persona afuera de algunas casas
            if index_seed % 4 == 0:
                glPushMatrix()
                glTranslatef(hx + 1.7, 0, hz - 1.2)
                draw_tree_round()
                glPopMatrix()
            elif index_seed % 6 == 0:
                glPushMatrix()
                glTranslatef(hx - 1.4, 0, hz + 1.0)
                color = colors_people[index_seed % len(colors_people)]
                draw_person(*color)
                glPopMatrix()

    # Gente caminando en banquetas / ciudad
    city_people = [
        (8, -8), (14, -12), (24, 21), (31, 19), (40, -28), (48, -24),
        (57, 36), (63, 34), (73, 5), (69, 10), (52, 54), (36, 56),
        (19, 40), (5, 27)
    ]
    for i, (px, pz) in enumerate(city_people):
        glPushMatrix()
        glTranslatef(px, 0, pz)
        draw_person(*colors_people[(i + 2) % len(colors_people)])
        glPopMatrix()

    # Perros extra
    for px, pz in [(18, 18), (62, 50)]:
        glPushMatrix()
        glTranslatef(px, 0.3, pz)
        draw_dog()
        glPopMatrix()


# ── Lógica de cámara / manos ──────────────────────────────────
def draw_hand_overlay(frame, keypoints, pinch_dist, handedness):
    color_node = (255, 120, 0) if handedness == "Right" else (0, 120, 255)

    for pt in keypoints:
        cv2.circle(frame, pt, 4, color_node, cv2.FILLED)

    for c in HAND_CONNECTIONS:
        cv2.line(frame, keypoints[c[0]], keypoints[c[1]], (80, 255, 80), 2)

    if handedness == "Right" and len(keypoints) >= 21:
        thumb, index = keypoints[4], keypoints[8]
        cv2.line(frame, thumb, index, (0, 255, 255), 2)
        mid = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
        cv2.putText(
            frame,
            f"Zoom: {int(pinch_dist)}px",
            mid,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )


def process_hands(results, w, h):
    global angle_x, angle_y, zoom, target_zoom, pan_x, pan_y
    global prev_right_index, prev_left_index

    if not results.hand_landmarks:
        prev_right_index = None
        prev_left_index = None
        return None

    hand_summary = []

    for idx, hand_lm in enumerate(results.hand_landmarks):
        handedness = "Left"
        if results.handedness and idx < len(results.handedness):
            handedness = results.handedness[idx][0].category_name

        keypoints = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
        if len(keypoints) < 21:
            continue

        thumb_tip = keypoints[4]
        index_tip = keypoints[8]
        pinch = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

        norm_x = index_tip[0] / w
        norm_y = index_tip[1] / h

        if handedness == "Right":
            # rotación
            if prev_right_index is not None:
                dx = (norm_x - prev_right_index[0]) * 150.0
                dy = (norm_y - prev_right_index[1]) * 150.0
                angle_y += dx
                angle_x = max(10.0, min(88.0, angle_x + dy))

            # zoom más amplio con el pinch
            calculated_zoom = -145.0 + (pinch / w) * 320.0
            target_zoom = max(MIN_ZOOM, min(MAX_ZOOM, calculated_zoom))
            prev_right_index = (norm_x, norm_y)

        else:
            # paneo con la mano izquierda
            if prev_left_index is not None:
                dx = (norm_x - prev_left_index[0]) * 70.0
                dy = (norm_y - prev_left_index[1]) * 70.0
                pan_x += dx
                pan_y -= dy
                pan_x = max(-40.0, min(40.0, pan_x))
                pan_y = max(-25.0, min(25.0, pan_y))
            prev_left_index = (norm_x, norm_y)

        hand_summary.append((keypoints, pinch, handedness))

    return hand_summary


# ── Main ──────────────────────────────────────────────────────
def main():
    global zoom

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Falta el archivo '{MODEL_PATH}' en este directorio.\n")
        sys.exit(1)

    if not glfw.init():
        sys.exit(1)

    win_w, win_h = 1280, 900
    window = glfw.create_window(win_w, win_h, "Metropoli Completa 3D Mejorada", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)
    glViewport(0, 0, win_w, win_h)

    glClearColor(0.05, 0.09, 0.15, 1.0)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(55, win_w / win_h, 0.3, 600.0)
    glMatrixMode(GL_MODELVIEW)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.55,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with HandLandmarker.create_from_options(options) as landmarker:
        while not glfw.window_should_close(window):
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            f_h, f_w, _ = frame.shape

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            results = landmarker.detect(mp_image)
            hand_data = process_hands(results, f_w, f_h)

            if hand_data:
                for pts, p_dist, side in hand_data:
                    draw_hand_overlay(frame, pts, p_dist, side)

            # suavizar zoom para que no se vea brusco
            zoom = zoom * 0.84 + target_zoom * 0.16

            cv2.imshow("Camara de Gestos", frame)
            cv2.waitKey(1)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            cam_z = -zoom
            gluLookAt(
                pan_x, cam_z * 0.62 + pan_y, cam_z,
                pan_x, 2.0 + pan_y, 0.0,
                0.0, 1.0, 0.0
            )

            glRotatef(angle_x, 1, 0, 0)
            glRotatef(angle_y, 0, 1, 0)

            draw_scenery()

            glfw.swap_buffers(window)
            glfw.poll_events()

    cap.release()
    cv2.destroyAllWindows()
    glfw.terminate()


if __name__ == "__main__":
    main()