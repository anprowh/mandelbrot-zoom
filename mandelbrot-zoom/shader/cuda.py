import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import keyboard as kb
from io import BytesIO

# define the Mandelbrot set kernel
mandelbrot_kernel = """
    #define MAX_ITERS 500
    
    __device__ int mandelbrot(double zr, double zi, double cr, double ci) {
        for (int i = 0; i < MAX_ITERS; i++) {
            double zr_new = zr * zr - zi * zi + cr;
            double zi_new = 2 * zr * zi + ci;
            zr = zr_new;
            zi = zi_new;
            if (zr * zr + zi * zi > 4.0)
                return i;
        }
        return MAX_ITERS;
    }
    
    __global__ void compute_mandelbrot(int* output, double xmin, double xmax, double ymin, double ymax, double cr, double ci, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < width && y < height) {
            double zr = xmin + (xmax - xmin) * x / width;
            double zi = ymin + (ymax - ymin) * y / height;
            int index = y * width + x;
            output[index] = mandelbrot(zr, zi, cr, ci);
        }
    }
"""

# compile the kernel
mod = SourceModule(mandelbrot_kernel)

# get the kernel function
mandelbrot_kernel = mod.get_function("compute_mandelbrot")

# set the image size
width = 512
height = 512

# set the zoom and position of the image
xmin = -2.0
xmax = 1.0
ymin = -1.5
ymax = 1.5

# allocate memory on the GPU for the output image
output_gpu = cuda.mem_alloc(width * height * np.dtype(np.int32).itemsize)

# compute the grid and block sizes
block_size = (16, 16, 1)
grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1], 1)

# copy the output image from the GPU to the CPU
output_cpu = np.empty((height, width), dtype=np.int32)

def smth(x, y, size):
    global output_cpu, output_gpu, block_size, grid_size, mandelbrot

    # run the kernel on the GPU
    mandelbrot_kernel(output_gpu, np.double(x), np.double(x+size), np.double(y), np.double(y+size), np.double(-0.5798738582286241), np.double(-0.4886769231743427), np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(output_cpu, output_gpu)

    output = (output_cpu - output_cpu.min()) / (output_cpu.max() - output_cpu.min()) * 255.0

    return Image.fromarray(np.repeat(output.astype(np.uint8)[:,:,np.newaxis], 3, 2), mode='HSV').convert(mode='RGB')

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

# # plot the image using matplotlib
# import matplotlib.pyplot as plt
# plt.imshow(output_cpu, cmap='hot', extent=(xmin, xmax, ymin, ymax))
# plt.show()