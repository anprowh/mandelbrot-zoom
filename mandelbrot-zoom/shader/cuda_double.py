import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt

# define the double-double arithmetic kernel
dd_kernel = """
    #define DD_SUM(a, b, c) do { \
        double t1 = (a) + (b); \
        double e = t1 - (a); \
        double t2 = ((b) - e) + ((a) - (t1)); \
        (c)[0] = t1; \
        (c)[1] = t2; \
    } while (0)

    __device__ void dd_mul(double a[2], double b[2], double c[2]) {
        double hi, lo;
        hi = a[0] * b[0];
        lo = a[0] * b[1] + a[1] * b[0];
        DD_SUM(hi, lo, c);
    }

    __device__ void dd_add(double a[2], double b[2], double c[2]) {
        double hi, lo;
        hi = a[0] + b[0];
        lo = a[1] + b[1];
        DD_SUM(hi, lo, c);
    }

    __device__ void dd_sub(double a[2], double b[2], double c[2]) {
        double hi, lo;
        hi = a[0] - b[0];
        lo = a[1] - b[1];
        DD_SUM(hi, lo, c);
    }

    __device__ void dd_div(double a[2], double b[2], double c[2]) {
        double q, r, t;
        q = a[0] / b[0];
        r = a[0] - q * b[0];
        t = (a[1] - q * b[1] - r * (b[1] / b[0])) / b[0];
        DD_SUM(q, t, c);
    }

    #define MAX_ITERS 1000

    __device__ int mandelbrot(double zr[2], double zi[2], double cr[2], double ci[2]) {
        for (int i = 0; i < MAX_ITERS; i++) {
            double zr2[2], zi2[2];
            dd_mul(zr, zr, zr2);
            dd_mul(zi, zi, zi2);
            dd_sub(zr2, zi2, zr2);
            dd_add(zr2, cr, zr2);
            dd_mul(zr, zi, zi2);
            dd_mul(zr, zi2, zi2);
            dd_add(zi2, ci, zi2);
            dd_add(zr2, zi2, zr);
            dd_add(zr2, zi2, zr2);
            if (zr[0] * zr[0] + zi[0] * zi[0] > 4.0) {
                return i;
            }
        }
        return MAX_ITERS;
    }

    __global__ void mandelbrot_set(int *output, int width, int height,
                                double xmin, double xmax, double ymin, double ymax,
                                double cr[2], double ci[2]) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }
        double zr[2], zi[2];
        double dx = (xmax - xmin) / width;
        double dy = (ymax - ymin) / height;
        zr[0] = xmin + x * dx;
        zi[0] = ymin + y * dy;
        zr[1] = zi[1] = 0.0;
        output[y * width + x] = mandelbrot(zr, zi, cr, ci);
    }

"""

# compile the kernels

dd_module = SourceModule(dd_kernel)

# get the kernel functions
mandelbrot_set = dd_module.get_function("mandelbrot_set")

# define the parameters of the Mandelbrot set computation

width = 800
height = 600
xmin = -2.0
xmax = 1.0
ymin = -1.5
ymax = 1.5
cr = np.array([0.0, 0.0])
ci = np.array([0.0, 0.0])
# allocate memory on the GPU for the output array

output_gpu = cuda.mem_alloc(width * height * np.dtype(np.int32).itemsize)
# compute the Mandelbrot set on the GPU

block_size = (16, 16, 1)
grid_size = ((width + block_size[0] - 1) // block_size[0],
(height + block_size[1] - 1) // block_size[1], 1)
mandelbrot_set(output_gpu, np.int32(width), np.int32(height),
np.float64(xmin), np.float64(xmax), np.float64(ymin), np.float64(ymax),
cuda.In(cr), cuda.In(ci), block=block_size, grid=grid_size)
# copy the output array from the GPU to the host

output_cpu = np.empty((height, width), dtype=np.int32)
cuda.memcpy_dtoh(output_cpu, output_gpu)
# plot the Mandelbrot set

{}.g

plt.imshow(output_cpu, cmap='jet', extent=[xmin, xmax, ymin, ymax])
plt.title(f"Mandelbrot set with c = ({cr[0]}, {ci[0]})")
plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.show()