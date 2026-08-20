
import math
import torch
import matplotlib.pyplot as plt


# Use GPU when CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Three vertices of an equilateral triangle
vertices = torch.tensor(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, math.sqrt(3) / 2]
    ],
    dtype=torch.float32,
    device=device
)

# Begin at the centre of the triangle
points = vertices.mean(dim=0, keepdim=True)

iterations = 11

# At each iteration, create three half-sized copies
for _ in range(iterations):
    points = (
        (points[:, None, :] + vertices[None, :, :]) / 2
    ).reshape(-1, 2)

# Move the final result to CPU for Matplotlib
points = points.cpu().numpy()

plt.figure(figsize=(10, 9))

plt.scatter(
    points[:, 0],
    points[:, 1],
    s=0.15,
    c=points[:, 1],
    cmap="viridis",
    marker="."
)

plt.title("Sierpinski Gasket")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    "sierpinski_gasket.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Device:", device)
print("Number of points:", len(points))
