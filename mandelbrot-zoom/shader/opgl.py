import glfw
from OpenGL.GL import *

def main():
    # Initialize GLFW and create a window
    if not glfw.init():
        return
    window = glfw.create_window(640, 480, "PyOpenGL Example", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)

    # Set up the viewport
    glViewport(0, 0, 640, 480)

    # Set up the projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 640, 0, 480, -1, 1)

    # Set up the modelview matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Set up the color
    glColor3f(1, 0, 0)

    # Draw the square
    glBegin(GL_QUADS)
    glVertex2f(100, 100)
    glVertex2f(100, 200)
    glVertex2f(200, 200)
    glVertex2f(200, 100)
    glEnd()

    # Swap buffers and poll events
    glfw.swap_buffers(window)
    while not glfw.window_should_close(window):
        glfw.poll_events()

    # Clean up
    glfw.terminate()

if __name__ == '__main__':
    main()
