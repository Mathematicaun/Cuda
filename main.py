import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import cupy as cp

def GPU(size, ra):
    start = time.time()
    for _ in range(ra):
        a = torch.rand(size, size).to('cuda')
        b = torch.rand(size, size).to('cuda')
        c = torch.matmul(a, b)
    return time.time()-start

def CPU(size, ra):
    start = time.time()
    for _ in range(ra):
        a = torch.rand(size, size).to('cpu')
        b = torch.rand(size, size).to('cpu')
        c = torch.matmul(a, b)
    return time.time()-start

def CUPY_F(size, ra):
    start = time.time()
    for _ in range(ra):
        a = cp.random.rand(size, size)
        b = cp.random.rand(size, size)
        c = cp.matmul(a, b)
    return time.time()-start

def NUMPY_F(size, ra):
    start = time.time()
    for _ in range(ra):
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.matmul(a, b)
    return time.time()-start


plt.rcParams['text.usetex'] = True
fig = plt.figure()
fig.set_facecolor('black')
k = 4
for i in range(k):
    ra = i+1
    size = np.arange(10**2, 10**3+10**2, 10**2)

    GPU_TIME = []
    CPU_TIME = []
    CUPY = []
    NUMPY = []
    for _ in size:
        GPU_TIME.append(GPU(_, ra))
    for _ in size:
        CPU_TIME.append(CPU(_, ra))
    for _ in size:
        CUPY.append(CUPY_F(_, ra))
    for _ in size:
        NUMPY.append(NUMPY_F(_, ra))

    ax = fig.add_subplot(int(k/2), int(k/2), int(i)+1)
    ax.tick_params(direction='in', color='white', labelcolor='white')
    plt.gca().spines[:].set(color='white', linewidth=1)
    ax.set_facecolor('black')
    ax.plot(size, CPU_TIME, label='TORCH ON CPU')
    ax.scatter(size, CPU_TIME, s=3)

    ax.plot(size, GPU_TIME, label='TORCH ON GPU')
    ax.scatter(size, GPU_TIME, s=3)

    ax.plot(size, CUPY, label='CUPY')
    ax.scatter(size, CUPY, s=3)

    ax.plot(size, NUMPY, label='NUMPY')
    ax.scatter(size, NUMPY, s=3)

    ax.set_xlabel(f'$Tensor$ $Dimension(range={ra})$', color='white')
    ax.set_ylabel('$Time$ $Needed(second)$', color='white')
    le = plt.legend()
    le.get_frame().set(color='none')
    for i in range(k):
        le.get_texts()[i].set_color('white')

plt.tight_layout()
plt.show()
