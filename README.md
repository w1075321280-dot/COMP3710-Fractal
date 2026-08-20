
# COMP3710 Fractal Implementation

This repository contains a PyTorch implementation of the Sierpinski Gasket.

## Algorithm

The program begins with a point at the centre of an equilateral triangle.
During each iteration, three half-sized copies of the existing points are
created and moved towards the three vertices.

After 11 iterations, the program generates 177,147 points.

## Requirements

- Python 3
- PyTorch
- Matplotlib

## Run

```bash
python fractal.py
