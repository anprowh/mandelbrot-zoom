import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def draw_escape(escape: np.ndarray):
    img = Image.fromarray(cp.asnumpy(escape))
    return img

def draw_mandelbrot(x_min,x_max,y_min,y_max, x_size=500, iters=256):
    y_size = int(x_size * (y_max-y_min) / (x_max-x_min))
    rangx = np.linspace(x_min, x_max, x_size)
    rangy = np.linspace(y_min, y_max, y_size)
    space = rangy[:,None]*1j + rangx
    
    space = cp.array(space)
    space1 = space
    escape_space = cp.zeros_like(space, np.uint8)
    first = -1
    for i in range(iters):
        if escape_space.sum()!=0 and first==-1:
            first = i
            escape_space[(escape_space==0) & (cp.imag(space1)**2 + cp.real(space1)**2 > 4)] = 1
        escape_space[(escape_space==0) & (cp.imag(space1)**2 + cp.real(space1)**2 > 4)] = min(255, i-first)
        
        space1 = space1 * space1 + space

    return draw_escape(escape_space)

def draw_mandelbrot2(x,y,size, x_size=500, iters=256):
    rangx = cp.linspace(x, x+size, x_size, dtype=cp.float64)
    rangy = cp.linspace(y, y+size, x_size, dtype=cp.float64)
    space = rangy[:,None]*1j + rangx
    
    space1 = space
    escape_space = cp.zeros_like(space, cp.uint8)
    first = -1

    for i in range(iters):
        if escape_space.sum()!=0 and first==-1:
            first = i
            escape_space[(escape_space==0) & (cp.imag(space1)**2 + cp.real(space1)**2 > 4)] = 1
        escape_space[(escape_space==0) & (cp.imag(space1)**2 + cp.real(space1)**2 > 4)] = min(255, i-first)
        
        space1 = space1 ** 2 + space

    escape_space[(escape_space==0)] = 255
    return draw_escape(escape_space)


def draw_mandelbrot3(x,y,size, x_size=500, iters=256):
    rangx = cp.linspace(x, x+size, x_size, dtype=cp.float64)
    rangy = cp.linspace(y, y+size, x_size, dtype=cp.float64)
    spacere = rangx.repeat(x_size).reshape(x_size, x_size).T
    spaceim = rangy.repeat(x_size).reshape(x_size, x_size)
    
    spacere1 = spacere
    spaceim1 = spaceim
    escape_space = cp.zeros_like(spacere, cp.uint8)
    first = -1

    for i in range(iters):
        if first == -1 and escape_space.sum()!=0:
            first = i
            escape_space[(escape_space==0) & (spacere1**2 + spaceim1**2 > 4)] = 1

        escape_space[(escape_space==0) & (spacere1**2 + spaceim1**2 > 4)] = min(255, i-first)

        spacere1, spaceim1 = spacere1 * spacere1 - spaceim1 * spaceim1 + spacere, 2.0 * spacere1 * spaceim1 + spaceim
    
    escape_space[escape_space==0] = 255
    return draw_escape(escape_space)

