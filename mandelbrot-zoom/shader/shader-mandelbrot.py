from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import * 
from OpenGL.GLU import * 
from OpenGL.GLUT import * 
import numpy as np

# Set the window size
WIDTH = 800
HEIGHT = 600

# Set the maximum number of iterations to determine if a point is in the Mandelbrot set
MAX_ITERATIONS = 100

# Define the shader source code
VERTEX_SHADER = """
#version 330
in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
out vec4 outputColor;

uniform float zoom;
uniform vec2 offset;

void main() {
    // Compute the position in the complex plane
    float x = (gl_FragCoord.x - offset.x) / zoom;
    float y = (gl_FragCoord.y - offset.y) / zoom;
    float zx = 0.0;
    float zy = 0.0;
    int i;
    for (i = 0; i < %d; i++) {
        float tmp = zx * zx - zy * zy + x;
        zy = 2.0 * zx * zy + y;
        zx = tmp;
        if (zx * zx + zy * zy > 4.0) {
            break;
        }
    }
    // Map the number of iterations to a color
    float c = float(i) / float(%d);
    outputColor = vec4(c, c, c, 1.0);
}
""" % (MAX_ITERATIONS, MAX_ITERATIONS)


glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
glutInitWindowSize(WIDTH, HEIGHT)
glutInitWindowPosition(50, 50)

glutInit(sys.argv)

glutCreateWindow(b"Mandelbrot")
glutMainLoop()

# # Compile the shader program
# shaderProgram = compileProgram(
#     compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
#     compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
# )

# # Set up the vertex buffer
# vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
# vertexBuffer = glGenBuffers(1)
# glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
# glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# # Set up the vertex attribute
# position = glGetAttribLocation(shaderProgram, "position")
# glEnableVertexAttribArray(position)
# glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

# # Set up the uniform variables
# zoom = glGetUniformLocation(shaderProgram, "zoom")
# offset = glGetUniformLocation(shaderProgram, "offset")

# # Start the main loop
# while True:

#     # Clear the screen
#     glClear(GL_COLOR_BUFFER_BIT)

#     # Set the uniform variables
#     glUniform1f(zoom, 1.0)
#     glUniform2f(offset, 0.0, 0.0)

#     # Draw the Mandelbrot set
#     glUseProgram(shaderProgram)
#     glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)