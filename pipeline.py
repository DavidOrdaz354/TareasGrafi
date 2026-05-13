import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import math

rotation_angle = 0.0

def draw_sphere():
    glColor3f(1.0, 0.2, 0.2)
    q = gluNewQuadric()
    gluSphere(q, 0.5, 32, 32)
    gluDeleteQuadric(q)

def draw_cube():
    glColor3f(0.2, 1.0, 0.2)
    glBegin(GL_QUADS)

    # Cara frontal
    glVertex3f(-0.4, -0.4, 0.4)
    glVertex3f(0.4, -0.4, 0.4)
    glVertex3f(0.4, 0.4, 0.4)
    glVertex3f(-0.4, 0.4, 0.4)

    # Cara trasera
    glVertex3f(-0.4, -0.4, -0.4)
    glVertex3f(-0.4, 0.4, -0.4)
    glVertex3f(0.4, 0.4, -0.4)
    glVertex3f(0.4, -0.4, -0.4)

    # Izquierda
    glVertex3f(-0.4, -0.4, -0.4)
    glVertex3f(-0.4, -0.4, 0.4)
    glVertex3f(-0.4, 0.4, 0.4)
    glVertex3f(-0.4, 0.4, -0.4)

    # Derecha
    glVertex3f(0.4, -0.4, -0.4)
    glVertex3f(0.4, 0.4, -0.4)
    glVertex3f(0.4, 0.4, 0.4)
    glVertex3f(0.4, -0.4, 0.4)

    # Arriba
    glVertex3f(-0.4, 0.4, -0.4)
    glVertex3f(-0.4, 0.4, 0.4)
    glVertex3f(0.4, 0.4, 0.4)
    glVertex3f(0.4, 0.4, -0.4)

    # Abajo
    glVertex3f(-0.4, -0.4, -0.4)
    glVertex3f(0.4, -0.4, -0.4)
    glVertex3f(0.4, -0.4, 0.4)
    glVertex3f(-0.4, -0.4, 0.4)

    glEnd()

def draw_cone():
    glColor3f(0.2, 0.2, 1.0)
    q = gluNewQuadric()
    glRotatef(-90, 1, 0, 0)
    gluCylinder(q, 0.5, 0.0, 1.0, 32, 32)
    gluDeleteQuadric(q)

def draw_cylinder():
    glColor3f(0.2, 1.0, 1.0)
    q = gluNewQuadric()
    glRotatef(-90, 1, 0, 0)
    gluCylinder(q, 0.4, 0.4, 1.0, 32, 32)
    gluDeleteQuadric(q)

def draw_disk():
    glColor3f(1.0, 0.5, 0.2)
    q = gluNewQuadric()
    gluDisk(q, 0.2, 0.6, 32, 32)
    gluDeleteQuadric(q)

def draw_partial_disk():
    glColor3f(0.8, 0.3, 0.8)
    q = gluNewQuadric()
    gluPartialDisk(q, 0.2, 0.6, 32, 16, 0, 270)
    gluDeleteQuadric(q)

def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glLightfv(GL_LIGHT0, GL_POSITION, [2,2,2,1])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3,0.3,0.3,1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1,1,1,1])

def draw_all():
    global rotation_angle

    shapes = [
        draw_sphere,
        draw_cube,
        draw_cone,
        draw_cylinder,
        draw_disk,
        draw_partial_disk
    ]

    cols = 3
    rows = 2

    width, height = glfw.get_window_size(glfw.get_current_context())

    for i, func in enumerate(shapes):
        col = i % cols
        row = i // cols

        w = width // cols
        h = height // rows

        glViewport(col*w, height-(row+1)*h, w, h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w/h, 0.1, 50)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0,0,3, 0,0,0, 0,1,0)

        glRotatef(rotation_angle, 1,1,0)

        glPushMatrix()
        func()
        glPopMatrix()

def main():
    global rotation_angle

    if not glfw.init():
        return

    window = glfw.create_window(1200, 800, "OpenGL sin GLUT", None, None)
    glfw.make_context_current(window)

    glClearColor(0.1,0.1,0.15,1)
    setup_lighting()

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rotation_angle += 0.5

        draw_all()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()