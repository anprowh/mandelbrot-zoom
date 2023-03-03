from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
# from render import draw_mandelbrot5, draw_escape
from io import BytesIO
import numpy as np
import socket

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

@app.get("/mandelbrot")
def get_mandelbrot(x: str, y: str, size: str):
    x = float(x)
    y = float(y)
    size = float(size)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(f"{x} {y} {size}".encode())
        data = s.recv(1024*1024*20)
    
    return Response(data, media_type="image/bmp")

# return html file 'index.html'
@app.get("/")
def root():
    return FileResponse('./index.html')