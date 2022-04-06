import time

import algorithms_cython as alc
import algorithms_jit as alj
import numpy as np

import algorithms as al
from PIL import Image, ImageDraw
from numba import njit


def draw_m(n):
    image = Image.new(mode="RGB", size=(n, n))

    draw = ImageDraw.Draw(image)

    w = image.size[0]
    h = image.size[1]
    pix = image.load()

    s = alc.mandelbrot_set(n)

    for x in range(s.shape[0]):
        for y in range(s.shape[1]):
            m = int(s[x, y]) * 5
            r = (4 * m) % 255
            g = (6 * m) % 255
            b = (8 * m) % 255
            draw.point((x, y), (r, g, b))

    image.show()
    image.save("res.png")

draw_m(2000)