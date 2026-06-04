import os
import sys
import math
import glfw
import cv2
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *

# Desactivar logs innecesarios de TensorFlow / Google
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

# -- MediaPipe Tasks API ──────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# -- Estado de la Camara 3D ───────────────────────────────────
angle_x, angle_y = 35.0, 45.0  
zoom = -45.0                   
pan_x, pan_y = 0.0, 0.0

prev_right_index = None
prev_left_index = None


# -- Componentes Primitivos de Renderizado ─────────────────────

def draw_generic_cube(w, h, d, r, g, b):
    """Dibuja un cubo apoyado en el suelo"""
    w_h, d_h = w / 2.0, d / 2.0
    glBegin(GL_QUADS)
    # Frente
    glColor3f(r, g, b)
    glVertex3f(-w_h, 0, d_h)
    glVertex3f(w_h, 0, d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(-w_h, h, d_h)
    # Atras
    glColor3f(r * 0.8, g * 0.8, b * 0.8)
    glVertex3f(-w_h, 0, -d_h)
    glVertex3f(w_h, 0, -d_h)
    glVertex3f(w_h, h, -d_h)
    glVertex3f(-w_h, h, -d_h)
    # Izquierda
    glColor3f(r * 0.7, g * 0.7, b * 0.7)
    glVertex3f(-w_h, 0, -d_h)
    glVertex3f(-w_h, 0, d_h)
    glVertex3f(-w_h, h, d_h)
    glVertex3f(-w_h, h, -d_h)
    # Derecha
    glColor3f(r * 0.7, g * 0.7, b * 0.7)
    glVertex3f(w_h, 0, -d_h)
    glVertex3f(w_h, 0, d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(w_h, h, -d_h)
    # Arriba
    glColor3f(r * 0.9, g * 0.9, b * 0.9)
    glVertex3f(-w_h, h, -d_h)
    glVertex3f(w_h, h, -d_h)
    glVertex3f(w_h, h, d_h)
    glVertex3f(-w_h, h, d_h)
    glEnd()


def draw_pyramid(w, h, d, r, g, b):
    """Primitiva piramidal"""
    w_h, d_h = w / 2.0, d / 2.0
    glBegin(GL_TRIANGLES)
    glColor3f(r, g, b)
    glVertex3f(-w_h, 0, d_h); glVertex3f(w_h, 0, d_h); glVertex3f(0, h, 0)
    glColor3f(r * 0.8, g * 0.8, b * 0.8)
    glVertex3f(-w_h, 0, -d_h); glVertex3f(w_h, 0, -d_h); glVertex3f(0, h, 0)
    glColor3f(r * 0.7, g * 0.7, b * 0.7)
    glVertex3f(-w_h, 0, -d_h); glVertex3f(-w_h, 0, d_h); glVertex3f(0, h, 0)
    glVertex3f(w_h, 0, -d_h); glVertex3f(w_h, 0, d_h); glVertex3f(0, h, 0)
    glEnd()


def draw_windows(w, h, d, rows, cols):
    """Crea la reticula de ventanas de edificios"""
    w_half, d_half = w / 2.0, d / 2.0
    win_w = w / (cols * 2)
    win_h = h / (rows * 2)
    
    glColor3f(0.95, 0.95, 0.4) 
    glBegin(GL_QUADS)
    z_f = d_half + 0.01
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 3 == 0: continue 
            x = -w_half + (c * (w / cols)) + win_w / 2
            y = (r * (h / rows)) + win_h / 2
            glVertex3f(x, y, z_f)
            glVertex3f(x + win_w, y, z_f)
            glVertex3f(x + win_w, y + win_h, z_f)
            glVertex3f(x, y + win_h, z_f)
    glEnd()


def draw_detailed_house(w, h, d, r, g, b):
    """Dibuja una casa habitacion"""
    draw_generic_cube(w, h, d, r, g, b)
    glPushMatrix()
    glTranslatef(0, 0.3, 0)
    draw_windows(w * 0.8, h * 0.6, d, 2, 2)
    glPopMatrix()
    
    w_h, d_h = (w + 0.2) / 2.0, (d + 0.2) / 2.0
    roof_y = h + 0.8
    
    glBegin(GL_TRIANGLES)
    glColor3f(0.85, 0.35, 0.1)
    glVertex3f(-w_h, h, d_h); glVertex3f(w_h, h, d_h); glVertex3f(0, roof_y, 0)
    glVertex3f(-w_h, h, -d_h); glVertex3f(w_h, h, -d_h); glVertex3f(0, roof_y, 0)
    glEnd()


# -- Estructuras Publicas e Infraestructura Urbana ─────────────

def draw_school():
    """Dibuja un edificio escolar grande en forma de L"""
    draw_generic_cube(12.0, 5.0, 5.0, 0.7, 0.7, 0.7)
    glPushMatrix()
    glTranslatef(4.0, 0, 4.0)
    draw_generic_cube(4.0, 5.0, 4.0, 0.7, 0.7, 0.7)
    glPopMatrix()
    draw_windows(10.0, 4.0, 5.0, 3, 5)
    glPushMatrix()
    glTranslatef(0, 5.0, 2.0)
    draw_generic_cube(2.0, 1.2, 0.5, 0.6, 0.1, 0.1)
    glPopMatrix()


def draw_church():
    """Dibuja una iglesia con torre campanario y cruz"""
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
    """Dibuja un parque pequeno con area infantil"""
    glBegin(GL_QUADS)
    glColor3f(0.35, 0.65, 0.3)
    glVertex3f(-7.0, 0.02, -7.0)
    glVertex3f(7.0, 0.02, -7.0)
    glVertex3f(7.0, 0.02, 7.0)
    glVertex3f(-7.0, 0.02, 7.0)
    glEnd()

    # Columpios
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

    # Resbaladilla
    glPushMatrix()
    glTranslatef(2.5, 0, -2.0)
    draw_generic_cube(0.8, 1.6, 0.8, 0.2, 0.4, 0.8)
    glPushMatrix()
    glTranslatef(0, 0.7, 1.2)
    glRotatef(30, 1, 0, 0)
    draw_generic_cube(0.7, 0.1, 2.2, 0.8, 0.8, 0.8)
    glPopMatrix()
    glPopMatrix()

    # Pasamanos
    glPushMatrix()
    glTranslatef(0, 0, 3.0)
    draw_generic_cube(0.1, 1.8, 0.1, 0.8, 0.6, 0.1)
    glTranslatef(0, 0, -4.0)
    draw_generic_cube(0.1, 1.8, 0.1, 0.8, 0.6, 0.1)
    glTranslatef(0, 1.8, 2.0)
    draw_generic_cube(0.12, 0.1, 4.2, 0.8, 0.6, 0.1)
    glPopMatrix()


def draw_kiosk():
    """Dibuja un kiosk urbano tradicional para plazas"""
    glPushMatrix()
    draw_generic_cube(3.2, 0.6, 3.2, 0.5, 0.35, 0.25) # Plataforma base de madera/cantera
    # Columnas delgadas periféricas
    glPushMatrix()
    glTranslatef(-1.4, 0.6, -1.4); draw_generic_cube(0.15, 2.2, 0.15, 0.8, 0.7, 0.5)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(1.4, 0.6, -1.4); draw_generic_cube(0.15, 2.2, 0.15, 0.8, 0.7, 0.5)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(1.4, 0.6, 1.4); draw_generic_cube(0.15, 2.2, 0.15, 0.8, 0.7, 0.5)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(-1.4, 0.6, 1.4); draw_generic_cube(0.15, 2.2, 0.15, 0.8, 0.7, 0.5)
    glPopMatrix()
    # Techo del kiosk
    glTranslatef(0, 2.8, 0)
    draw_pyramid(3.6, 1.5, 3.6, 0.7, 0.2, 0.2) # Techo piramidal color teja roja
    glPopMatrix()


def draw_street_lamp():
    """Dibuja un poste de luz urbana"""
    glPushMatrix()
    draw_generic_cube(0.2, 4.0, 0.2, 0.2, 0.2, 0.22)
    glTranslatef(0, 4.0, 0.3)
    draw_generic_cube(0.2, 0.2, 0.8, 0.2, 0.2, 0.22)
    glTranslatef(0, -0.1, 0.3)
    draw_generic_cube(0.4, 0.2, 0.4, 0.95, 0.95, 0.5)
    glPopMatrix()


def draw_traffic_light():
    """Dibuja un semaforo vial"""
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


def draw_dog():
    """Dibuja una mascota usando bloques minimalistas"""
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


# -- Modelado de Flota Vehicular ───────────────────────────────

def draw_truck():
    """Dibuja un camion de transporte pesado"""
    glPushMatrix()
    draw_generic_cube(1.6, 1.8, 4.0, 0.85, 0.85, 0.85)
    glTranslatef(0, 0, 2.3)
    draw_generic_cube(1.5, 1.1, 1.2, 0.8, 0.1, 0.1)
    glPopMatrix()


def draw_motorcycle(r, g, b):
    """Dibuja una motocicleta deportiva"""
    glPushMatrix()
    draw_generic_cube(0.4, 0.6, 1.2, r, g, b)
    glTranslatef(0, -0.2, 0.5)
    draw_generic_cube(0.2, 0.3, 0.3, 0.05, 0.05, 0.05)
    glTranslatef(0, 0, -1.0)
    draw_generic_cube(0.2, 0.3, 0.3, 0.05, 0.05, 0.05)
    glPopMatrix()


def draw_car(r, g, b):
    """Dibuja un automovil convencional"""
    glPushMatrix()
    draw_generic_cube(1.1, 0.4, 2.0, r, g, b)
    glPushMatrix()
    glTranslatef(0, 0.4, -0.1)
    draw_generic_cube(0.9, 0.35, 1.1, r * 0.6, g * 0.6, b * 0.6)
    glPopMatrix()
    draw_generic_cube(1.25, 0.25, 0.4, 0.05, 0.05, 0.05)
    glPushMatrix(); glTranslatef(0, 0, 1.1); draw_generic_cube(1.25, 0.25, 0.4, 0.05, 0.05, 0.05); glPopMatrix()
    glPopMatrix()


# -- Modelado del Complejo Deportivo ───────────────────────────

def draw_volleyball_court():
    """Dibuja una cancha de voleibol"""
    glBegin(GL_QUADS)
    glColor3f(0.9, 0.5, 0.2)
    glVertex3f(-4.0, 0.02, -7.0); glVertex3f(4.0, 0.02, -7.0); glVertex3f(4.0, 0.02, 7.0); glVertex3f(-4.0, 0.02, 7.0)
    glColor3f(0.1, 0.4, 0.7)
    glVertex3f(-3.0, 0.025, -6.0); glVertex3f(3.0, 0.025, -6.0); glVertex3f(3.0, 0.025, 6.0); glVertex3f(-3.0, 0.025, 6.0)
    glEnd()
    glLineWidth(2.0); glBegin(GL_LINES); glColor3f(1.0, 1.0, 1.0); glVertex3f(-3.0, 0.03, 0.0); glVertex3f(3.0, 0.03, 0.0); glEnd()
    draw_generic_cube(0.1, 2.2, 0.1, 0.6, 0.6, 0.6)
    glPushMatrix(); glTranslatef(0, 1.3, 0); draw_generic_cube(6.0, 0.7, 0.02, 0.9, 0.9, 0.9); glPopMatrix()


def draw_basketball_court():
    """Dibuja una cancha de basquetbol"""
    glBegin(GL_QUADS)
    glColor3f(0.75, 0.52, 0.3)
    glVertex3f(-4.0, 0.02, -8.0); glVertex3f(4.0, 0.02, -8.0); glVertex3f(4.0, 0.02, 8.0); glVertex3f(-4.0, 0.02, 8.0)
    glEnd()
    glPushMatrix(); glTranslatef(0, 0, -7.5); draw_generic_cube(0.15, 3.0, 0.15, 0.2, 0.2, 0.2); glTranslatef(0, 3.0, 0.2); draw_generic_cube(1.8, 1.1, 0.05, 1.0, 1.0, 1.0); glTranslatef(0, -0.3, 0.2); draw_generic_cube(0.5, 0.1, 0.5, 0.9, 0.1, 0.1); glPopMatrix()
    glPushMatrix(); glTranslatef(0, 0, 7.5); draw_generic_cube(0.15, 3.0, 0.15, 0.2, 0.2, 0.2); glTranslatef(0, 3.0, -0.2); draw_generic_cube(1.8, 1.1, 0.05, 1.0, 1.0, 1.0); glTranslatef(0, -0.3, -0.2); draw_generic_cube(0.5, 0.1, 0.5, 0.9, 0.1, 0.1); glPopMatrix()


# -- Vegetacion Urbana Procedural ──────────────────────────────

def draw_tree_round():
    draw_generic_cube(0.25, 1.0, 0.25, 0.4, 0.25, 0.15)
    glPushMatrix(); glTranslatef(0, 1.0, 0); draw_generic_cube(1.2, 1.1, 1.2, 0.2, 0.55, 0.2); glPopMatrix()

def draw_tree_square():
    draw_generic_cube(0.25, 0.8, 0.25, 0.35, 0.2, 0.1)
    glPushMatrix(); glTranslatef(0, 0.8, 0); draw_generic_cube(0.9, 1.4, 0.9, 0.15, 0.45, 0.15); glPopMatrix()

def draw_tree_pine():
    draw_generic_cube(0.25, 0.7, 0.25, 0.4, 0.25, 0.15)
    glPushMatrix()
    glTranslatef(0, 0.6, 0); draw_pyramid(1.5, 1.0, 1.5, 0.1, 0.38, 0.15)
    glTranslatef(0, 0.6, 0); draw_pyramid(1.1, 0.9, 1.1, 0.12, 0.42, 0.18)
    glTranslatef(0, 0.5, 0); draw_pyramid(0.7, 0.7, 0.7, 0.15, 0.48, 0.22)
    glPopMatrix()

def draw_tree_double_sphere():
    draw_generic_cube(0.2, 1.8, 0.2, 0.4, 0.25, 0.15)
    glPushMatrix()
    glTranslatef(0, 0.8, 0); draw_generic_cube(0.9, 0.7, 0.9, 0.25, 0.6, 0.25)
    glTranslatef(0, 0.7, 0); draw_generic_cube(0.7, 0.5, 0.7, 0.3, 0.65, 0.3)
    glPopMatrix()


# -- Construccion Escenografica de la Metropoli ────────────────

def draw_scenery():
    # Base del Terreno Urbano
    glBegin(GL_QUADS)
    glColor3f(0.14, 0.14, 0.15)
    glVertex3f(-55, -0.01, 55); glVertex3f(55, -0.01, 55); glVertex3f(55, -0.01, -55); glVertex3f(-55, -0.01, -55)
    glEnd()
    
    # Parque Principal
    glBegin(GL_QUADS)
    glColor3f(0.2, 0.42, 0.22)
    glVertex3f(-50, 0.01, 50); glVertex3f(-15, 0.01, 50); glVertex3f(-15, 0.01, -50); glVertex3f(-50, 0.01, -50)
    glEnd()
    
    # Arboles del parque principal
    for z in range(-45, 46, 7):
        glPushMatrix(); glTranslatef(-45, 0, z); draw_tree_round(); glPopMatrix()
        glPushMatrix(); glTranslatef(-38, 0, z); draw_tree_square(); glPopMatrix()
        if z < -15 or z > 15:
            glPushMatrix(); glTranslatef(-30, 0, z); draw_tree_pine(); glPopMatrix()
            glPushMatrix(); glTranslatef(-22, 0, z); draw_tree_double_sphere(); glPopMatrix()

    # Canchas Deportivas
    glPushMatrix(); glTranslatef(-26, 0, -8); draw_volleyball_court(); glPopMatrix()
    glPushMatrix(); glTranslatef(-26, 0, 10); draw_basketball_court(); glPopMatrix()

    # Kiosk central integrado en medio de las areas recreativas del parque grande
    glPushMatrix(); glTranslatef(-33.0, 0, -22.0); draw_kiosk(); glPopMatrix()

    # Edificios de Servicios Publicos
    glPushMatrix(); glTranslatef(34.0, 0, 42.0); draw_school(); glPopMatrix()
    glPushMatrix(); glTranslatef(42.0, 0, 12.0); draw_church(); glPopMatrix()

    # Parque Pequeno Infantil y Mascota
    glPushMatrix()
    glTranslatef(20.0, 0, -10.0)
    draw_small_playground()
    glTranslatef(-2.0, 0.3, 1.0)
    draw_dog()
    glPopMatrix()

    # Red Vial / Calles Principales
    glColor3f(0.28, 0.28, 0.3)
    glBegin(GL_QUADS)
    for i in range(-10, 55, 13):
        glVertex3f(i - 1.3, 0.015, 50); glVertex3f(i + 1.3, 0.015, 50); glVertex3f(i + 1.3, 0.015, -50); glVertex3f(i - 1.3, 0.015, -50)
        glVertex3f(-15, 0.015, i + 1.3); glVertex3f(50, 0.015, i + 1.3); glVertex3f(50, 0.015, i - 1.3); glVertex3f(-15, 0.015, i - 1.3)
    glEnd()

    # Infraestructura Vial: Postes de Luz y Semaforos
    glPushMatrix(); glTranslatef(12.0, 0, 15.0); draw_street_lamp(); glPopMatrix()
    glPushMatrix(); glTranslatef(25.0, 0, -12.0); draw_street_lamp(); glPopMatrix()
    glPushMatrix(); glTranslatef(12.0, 0, -12.0); draw_traffic_light(); glPopMatrix()
    glPushMatrix(); glTranslatef(25.0, 0, 15.0); draw_traffic_light(); glPopMatrix()

    # Trafico Vial Ordinario
    glPushMatrix(); glTranslatef(2, 0.1, -25); draw_car(0.85, 0.1, 0.1); glPopMatrix()
    glPushMatrix(); glTranslatef(15, 0.1, 5); draw_car(0.1, 0.3, 0.8); glPopMatrix()
    glPushMatrix(); glTranslatef(28, 0.1, 20); draw_car(0.9, 0.8, 0.1); glPopMatrix()
    glPushMatrix(); glTranslatef(41, 0.1, -10); draw_car(0.15, 0.6, 0.2); glPopMatrix()
    glPushMatrix(); glTranslatef(15, 0.1, -38); draw_car(0.95, 0.95, 0.95); glPopMatrix()
    glPushMatrix(); glTranslatef(2, 0.1, 15); draw_truck(); glPopMatrix()
    glPushMatrix(); glTranslatef(28, 0.15, -30); draw_motorcycle(0.1, 0.8, 0.8); glPopMatrix()
    glPushMatrix(); glTranslatef(41, 0.15, 5); draw_motorcycle(0.9, 0.1, 0.5); glPopMatrix()
    glPushMatrix(); glTranslatef(2, 0.1, -5); draw_car(0.9, 0.4, 0.0); glPopMatrix()
    glPushMatrix(); glTranslatef(15, 0.1, 32); draw_car(0.4, 0.1, 0.6); glPopMatrix()
    glPushMatrix(); glTranslatef(28, 0.1, -12); draw_car(0.1, 0.1, 0.1); glPopMatrix()
    glPushMatrix(); glTranslatef(41, 0.1, -42); draw_car(0.5, 0.35, 0.05); glPopMatrix()

    # Distrito Comercial de Rascacielos
    skyscrapers = [
        (2.0, -35, 4.5, 18.0, 4.5, 0.25, 0.35, 0.5),
        (15.0, -35, 4.0, 24.0, 4.0, 0.2, 0.2, 0.3),
        (28.0, -35, 5.0, 28.0, 5.0, 0.15, 0.4, 0.45),
        (41.0, -35, 4.2, 16.0, 4.2, 0.3, 0.3, 0.35),
        (2.0, -22, 4.0, 15.0, 4.0, 0.35, 0.35, 0.4),
        (15.0, -22, 5.5, 34.0, 5.5, 0.1, 0.25, 0.5),
    ]
    for x, z, w, h, d, r, g, b in skyscrapers:
        glPushMatrix()
        glTranslatef(x, 0, z)
        draw_generic_cube(w, h, d, r, g, b)
        draw_windows(w, h, d, int(h // 1.6), 4)
        glPopMatrix()

    # Zona Residencial Dinamica
    house_colors = [
        (0.85, 0.45, 0.45), (0.45, 0.65, 0.85), (0.55, 0.75, 0.55),
        (0.85, 0.80, 0.55), (0.75, 0.60, 0.80), (0.80, 0.80, 0.80)
    ]
    for hz in range(0, 35, 9):
        for hx in range(0, 35, 8):
            if hx in [16, 24] or hz in [18, 27]: continue
            
            index_seed = int((hx * 3 + hz * 7) % 100)
            w_var = 2.4 + (index_seed % 3) * 0.5      
            h_var = 2.0 + ((index_seed + 1) % 3) * 0.4 
            d_var = 2.4 + ((index_seed + 2) % 3) * 0.5 
            
            offset_x = ((index_seed % 4) - 2) * 0.4
            offset_z = (((index_seed + 2) % 4) - 2) * 0.4
            r_c, g_c, b_c = house_colors[index_seed % len(house_colors)]
            
            glPushMatrix()
            glTranslatef(hx + 1.5 + offset_x, 0, hz + 1.5 + offset_z)
            draw_detailed_house(w_var, h_var, d_var, r_c, g_c, b_c)
            glPopMatrix()


# -- Logica de Captura e Interaccion Gestual ───────────────────

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
        cv2.putText(frame, f"Zoom: {int(pinch_dist)}px", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


def process_hands(results, w, h):
    global angle_x, angle_y, zoom, pan_x, pan_y
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
        if len(keypoints) < 21: continue

        thumb_tip = keypoints[4]
        index_tip = keypoints[8]
        pinch = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

        norm_x = index_tip[0] / w
        norm_y = index_tip[1] / h

        if handedness == "Right":
            if prev_right_index is not None:
                dx = (norm_x - prev_right_index[0]) * 130.0
                dy = (norm_y - prev_right_index[1]) * 130.0
                angle_y += dx
                angle_x = max(15.0, min(85.0, angle_x + dy))
            
            calculated_zoom = -75.0 + (pinch / w) * 110.0
            zoom = max(-90.0, min(-15.0, calculated_zoom))
            prev_right_index = (norm_x, norm_y)
        else:
            if prev_left_index is not None:
                dx = (norm_x - prev_left_index[0]) * 50.0
                dy = (norm_y - prev_left_index[1]) * 50.0
                pan_x += dx
                pan_y -= dy
            prev_left_index = (norm_x, norm_y)

        hand_summary.append((keypoints, pinch, handedness))
    return hand_summary


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Falta el archivo '{MODEL_PATH}' en este directorio.\n")
        sys.exit(1)

    if not glfw.init():
        sys.exit(1)

    win_w, win_h = 1100, 850
    window = glfw.create_window(win_w, win_h, "Metropoli Completa 3D", None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)

    glfw.make_context_current(window)
    glViewport(0, 0, win_w, win_h)

    glClearColor(0.06, 0.1, 0.16, 1.0) 
    glEnable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, win_w / win_h, 1.0, 350.0)
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
            if not ret: break

            frame = cv2.flip(frame, 1)
            f_h, f_w, _ = frame.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            results = landmarker.detect(mp_image)
            hand_data = process_hands(results, f_w, f_h)

            if hand_data:
                for pts, p_dist, side in hand_data:
                    draw_hand_overlay(frame, pts, p_dist, side)

            cv2.imshow("Camara de Gestos", frame)
            cv2.waitKey(1)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            cam_z = -zoom
            gluLookAt(pan_x, cam_z * 0.6 + pan_y, cam_z, pan_x, 2.0 + pan_y, 0.0, 0.0, 1.0, 0.0)

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