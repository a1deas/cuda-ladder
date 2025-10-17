#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

// Parametres for fractal render
struct Params {
	int width = 1280, height = 720, max_iter = 500;
	double centerX = -0.75, centerY = 0.0;
	double scale = 1.5;
	bool julia = false;
	double juliaA = -0.8, juliaB = 0.156;
	std::string out = "out.ppm";
};

// Writes fractal ppm file
static void writePpm(const std::string& path, const std::vector<unsigned char>& img, int width, int height) {
	FILE* file = std::fopen(path.c_str(), "wb");
	if (!file) {
		std::perror("fopen");
		std::exit(1);
	}

	std::fprintf(file, "P6\n%d %d\n255\n", width, height);
	std::fwrite(img.data(), 1, img.size(), file);
	std::fclose(file);
}

// Applys rgba color to specific pixel
__host__ __device__ inline void colorFromT(float t, unsigned char& red, unsigned char& green, unsigned char& blue) {
	red = (unsigned char)(9 * (1 - t) * t * t * t * 255);
	green = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
	blue = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
}

__host__ __device__ inline float2 cmul(float2 a, float2 b) {
	return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Processes Mandelbrot/Julia set
__global__ void fractalKernel(uchar4* out, 
	int width, 
	int height, 
	float centerX, 
	float centerY, 
	float scale, 
	int max_iter, 
	int modeJulia, 
	float juliaA, 
	float juliaB) 
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) {
		return;
	}

	float aspect = (float)width / (float)height;
	float u = (((x + 0.5f) / width) - 0.5f) * 2.f * aspect * scale + centerX;
	float v = (((y + 0.5f) / height) - 0.5f) * 2.f * scale + centerY;

	float2 z = modeJulia ? make_float2(u, v) : make_float2(0.f, 0.f);
	float2 c = modeJulia ? make_float2(juliaA, juliaB) : make_float2(u, v);

	int n = 0;
	float rr = 0.f;
	for (; n < max_iter; ++n) {
		float2 z2 = cmul(z, z);
		z.x = z2.x + c.x;
		z.y = z2.y + c.y;
		rr = z.x * z.x + z.y * z.y;
		if (rr > 4.f) {
			break;
		}
	}

	float t = 0.f;
	if (n == max_iter) {
		t = 0.f;
	}
	else {
		t = n + 1 - log2f(logf(sqrtf(rr)));
		t /= (float)max_iter;
		if (t < 0.f) t = 0.f;
		if (t > 1.f) t = 1.f;
	}

	unsigned char red, green, blue;
	colorFromT(t, red, green, blue);
	out[y * width + x] = make_uchar4(red, green, blue, 255);
}

// CUDA Pipeline function, that launches kernel and generates fractal
static void renderCuda(const Params& params, std::vector<unsigned char>& rgb) {
	rgb.assign(params.width * params.height * 3, 0);
	uchar4* dOut = nullptr;
	cudaMalloc(&dOut, size_t(params.width) * params.height * sizeof(uchar4));

	// Initialise blocks and threads
	dim3 block(16, 16);
	dim3 grid((params.width + block.x - 1) / block.x, (params.height + block.y - 1) / block.y);

	fractalKernel<<<grid, block>>>(
		dOut, 
		params.width, 
		params.height, 
		(float)params.centerX, 
		(float)params.centerY, 
		(float)params.scale,
		params.max_iter, 
		params.julia ? 1 : 0,
		(float)params.juliaA, 
		(float)params.juliaB
		);
	cudaDeviceSynchronize();

	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
	}

	std::vector<uchar4> temp(size_t(params.width) * params.height);
	cudaMemcpy(temp.data(), dOut, temp.size() * sizeof(uchar4), cudaMemcpyDeviceToHost);
	cudaFree(dOut);

	for (size_t i = 0; i < temp.size(); ++i) {
		rgb[3 * i + 0] = temp[i].x;
		rgb[3 * i + 1] = temp[i].y;
		rgb[3 * i + 2] = temp[i].z;
	}
	
}

static void usage() {
	std::cout << 
		"Options:\n"
		"  --width W --height H\n"
		"  --center_x X --center_y Y --scale S\n"
		"  --max_iter N\n"
		"  --julia --julia_a A --julia_b B\n"
		"  --out FILE.ppm\n";
}

int main(int argc, char** argv) {
	Params params;

	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		auto need = [&](const char* name) { return (i + 1 < argc && a == name) ? std::string(argv[++i]) : std::string(); };
		if (a == "--width")       params.width = std::stoi(need("--width"));
		else if (a == "--height") params.height = std::stoi(need("--height"));
		else if (a == "--center_x") params.centerX = std::stod(need("--center_x"));
		else if (a == "--center_y") params.centerY = std::stod(need("--center_y"));
		else if (a == "--scale")    params.scale = std::stod(need("--scale"));
		else if (a == "--max_iter") params.max_iter = std::stoi(need("--max_iter"));
		else if (a == "--out")      params.out = need("--out");
		else if (a == "--julia")    params.julia = true;
		else if (a == "--julia_a")  params.juliaA = std::stod(need("--julia_a"));
		else if (a == "--julia_b")  params.juliaB = std::stod(need("--julia_b"));
		else { usage(); return 0; }
	}

	std::vector<unsigned char> img;
	renderCuda(params, img);
	writePpm(params.out, img, params.width, params.height);

	std::cerr << "Saved " << params.out << " (" << params.width << "x" << params.height << ")\n";
	return 0;
}