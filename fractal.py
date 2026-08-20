import math
fractal_dimension = math.log(3)/math.log(2)

print(
    "Theoretical fractal dimension:",
    fractal_dimension
)

import torch
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

width = 1025
steps = 512
grid = torch.zeros((steps,width),dtype=torch.int8,device=device)

grid[0,width//2] = 1
#XOR
for t in range(1, steps):
    prev = grid[t-1]
    next_row = torch.zeros_like(prev)
    next_row[1:-1] = torch.bitwise_xor(prev[:-2], prev[2:])
    grid[t] = next_row

image = grid.cpu().numpy() #cpu

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap="binary", interpolation="nearest", aspect="auto")
plt.title("Sierpinski Triangle")
plt.axis("off")
plt.tight_layout()

plt.savefig("sierpinski_rule90.png",dpi=300,bbox_inches="tight")
plt.show()

print("Device:", device)
print("Grid shape:", image.shape)
print("Alive cells:", image.sum())
