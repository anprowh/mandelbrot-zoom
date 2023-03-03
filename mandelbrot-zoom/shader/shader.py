import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
from PIL import Image
import keyboard as kb
from io import BytesIO

SAVE_IMAGE = True

VERTEX_SHADER = """
#version 330

layout(location = 0) in vec2 position;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330

uniform vec2 resolution;
uniform vec2 mouse;

out vec4 fragColor;

void main()
{
    vec2 position = gl_FragCoord.xy / resolution;
    vec3 color = vec3(abs(position-mouse/resolution), 1.0);
    fragColor = vec4(color, 1.0);
}
"""

MAX_ITERATIONS = 100

FRAGMENT_SHADER = """
#version 330
out vec4 outputColor;

uniform float size;
uniform vec2 offset;
uniform vec2 resolution;

void main() {
    // Compute the position in the complex plane
    float x = gl_FragCoord.x / resolution.y * size + offset.x;
    float y = gl_FragCoord.y / resolution.y * size + offset.y;
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

def create_shader_program(vertex_shader, fragment_shader):
    shader_program = glCreateProgram()

    # Compile vertex shader
    vertex_shader_id = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader_id, vertex_shader)
    glCompileShader(vertex_shader_id)
    if not glGetShaderiv(vertex_shader_id, GL_COMPILE_STATUS):
        raise RuntimeError("Vertex shader compilation failed: " + glGetShaderInfoLog(vertex_shader_id))

    # Compile fragment shader
    fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader_id, fragment_shader)
    glCompileShader(fragment_shader_id)
    if not glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS):
        raise RuntimeError("Fragment shader compilation failed: " + glGetShaderInfoLog(fragment_shader_id))

    # Attach shaders to program
    glAttachShader(shader_program, vertex_shader_id)
    glAttachShader(shader_program, fragment_shader_id)

    # Link program
    glLinkProgram(shader_program)
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        raise RuntimeError("Shader program linking failed: " + glGetProgramInfoLog(shader_program))

    # Delete shaders (no longer needed)
    glDeleteShader(vertex_shader_id)
    glDeleteShader(fragment_shader_id)

    return shader_program



# Initialize GLFW and create a window
glfw.init()
    

if SAVE_IMAGE:
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

window = glfw.create_window(640, 480, "PyOpenGL Example", None, None)
if not window:
    glfw.terminate()
    
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

# Create the shader program
shader_program = create_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
glUseProgram(shader_program)

if SAVE_IMAGE:
    # Set up the framebuffer
    frame_buffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer)

    # Set up a texture to render to
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    # Check framebuffer status
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("Framebuffer is not complete")


# Set up the vertex buffer
vertex_data = [
    -1, -1, -1, 1, 1, 1, 1, -1
]
vertex_buffer = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
glBufferData(GL_ARRAY_BUFFER, len(vertex_data) * 4, (GLfloat * len(vertex_data))(*vertex_data), GL_STATIC_DRAW)

# Set up the position attribute
position_location = glGetAttribLocation(shader_program, "position")
glEnableVertexAttribArray(position_location)
glVertexAttribPointer(position_location, 2, GL_FLOAT, GL_FALSE, 0, None)

# Set up the resolution uniform
resolution_location = glGetUniformLocation(shader_program, "resolution")
glUniform2f(resolution_location, 640, 480)

# Clear color buffer
glClearColor(0.0, 0.0, 0.0, 0.0)
glClear(GL_COLOR_BUFFER_BIT)

# Draw the rectangle (4 vertices)
glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

size_location = glGetUniformLocation(shader_program, "size")
offset_location = glGetUniformLocation(shader_program, "offset")

# Main loop
def smth(x, y, size): 
    global SAVE_IMAGE, frame_buffer, texture, shader_program, size_location, offset_location
    # Set up the mouse uniform
    # mouse_location = glGetUniformLocation(shader_program, "mouse")
    # x, y = glfw.get_cursor_pos(window)
    # glUniform2f(mouse_location, x, y)

    glUniform1f(size_location, size)

    glUniform2f(offset_location, x, y)

    # Draw the square
    glDrawArrays(GL_QUADS, 0, 4)


    if SAVE_IMAGE:
        # Render to the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer)
        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_QUADS, 0, 4)

        # Read pixel data from the framebuffer
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        pixel_data = glReadPixels(0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE)
        image_data = Image.frombytes('RGB', (640, 480), pixel_data)

        # Save the image to a file
        image_data.save('rendered_image.png')


    # Swap buffers and poll events
    glfw.swap_buffers(window)
    glfw.poll_events()
    return image_data


import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    while not kb.is_pressed('esc'):
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print(f"Received {data.decode('utf-8')}")
                data = data.decode('utf-8')
                if data == "exit": break
                img = smth(*(float(x) for x in data.split()))
                f = BytesIO()
                img.save(f, format='bmp')
                conn.sendall(f.getvalue())

# kb.wait('esc')



# Clean up
glDeleteBuffers(1, [vertex_buffer])
glDeleteProgram(shader_program)
glfw.terminate()