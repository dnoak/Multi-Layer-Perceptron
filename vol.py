import numpy as np
import numba

@numba.njit
def get_coords(n, r):
    coords = np.zeros(((2*n)**3, 3), dtype=np.float32)
    pos = 0
    for z in range(-n, n):
        z_shift = r*z * np.sqrt(2)
        for y in range(-n, n):
            y_shift = 2*r*y + r * (z % 2)
            for x in range(-n, n):
                x_shift = 2*r*x + r * (z % 2)
                coords[pos] = [x_shift, y_shift, z_shift]
                pos += 1
    return coords

@numba.njit
def is_inside(coords, rj, rt, samples):
    counts = np.zeros(samples, dtype=np.int32)
    for r in range(samples):
        xr, yr, zr = np.random.random(3) * 2
        for (x, y, z) in coords:
            if (x+xr)**2 + (y+yr)**2 + (z+zr)**2 < (rj/rt)**2:
                counts[r] += 1
    return set(sorted(counts))

rj = 69.911
rt =6.371

coords = get_coords(20, 1)
print(is_inside(coords, rj, rt, 100))

print(4*4/3*np.pi/(16*np.sqrt(2))*rj**3/rt**3)