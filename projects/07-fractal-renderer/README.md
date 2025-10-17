# 07 - Fractal Renderer(reduced)
> Full: https://github.com/a1deas/Fractal-Renderer

**Goal:**
Try generating my first fractal, test its performance and use knowledge in a separate Fractal Renderer project. 

**Files:**
- 'kernel.cu' - Base Fractal logic.
- 'img/fractal.ppm' - generated ppm.
- 'img/fracta.png' - mandelbrot demo.

**Concepts:**
- CUDA pipeline: **Host init → cudaMalloc → H→D copy → kernel launch → sync → D→H copy → check → free**  
- Mandelbrot/Julia set.
- First real case.