#include "window.h"

void Window::draw_triangle(Point a, Point b, Point c) {
    a = point_to_screen(a);
    b = point_to_screen(b);
    c = point_to_screen(c);

    Point* d_a;
    Point* d_b;
    Point* d_c;

    void* d_pixels;
    int* d_depth_buffer;

    cudaMalloc(&d_a, sizeof(Point));
    cudaMalloc(&d_b, sizeof(Point));
    cudaMalloc(&d_c, sizeof(Point));
    cudaMalloc(&d_pixels, surface->pitch * surface->h);
    cudaMalloc(&d_depth_buffer, surface->w * surface->h * sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pixels, surface->pixels, surface->pitch * surface->h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_depth_buffer, depth_buffer, surface->w * surface->h * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
    draw_triangle_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, d_pixels, surface->pitch, surface->w, surface->h, d_depth_buffer);

    cudaMemcpy(surface->pixels, d_pixels, surface->pitch * surface->h, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_pixels);
    cudaFree(d_depth_buffer);
}

__global__ void draw_triangle_kernel(Point* a, Point* b, Point* c, void* pixels, int pitch, int width, int height, int* depth_buffer) {
    float x = blockIdx.x * blockDim.x + threadIdx.x;
    float y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    Point p = {x, y, 0, 255, 255, 255};

    float area = edge_function(a, b, c);
    float w0 = edge_function(b, c, &p) / area;
    float w1 = edge_function(c, a, &p) / area;
    float w2 = edge_function(a, b, &p) / area;

    if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
        p.r = w0 * a->r + w1 * b->r + w2 * c->r;
        p.g = w0 * a->g + w1 * b->g + w2 * c->g;
        p.b = w0 * a->b + w1 * b->b + w2 * c->b;

        p.z = w0 * a->z + w1 * b->z + w2 * c->z;

        if (p.z > depth_buffer[(int) y * width + (int) x]) {
            return;
        }

        Uint8* pixel = (Uint8*) pixels;
        pixel += ((int) y * pitch) + ((int) x * sizeof(Uint32));
        pixel[2] = p.r;
        pixel[1] = p.g;
        pixel[0] = p.b;
        depth_buffer[(int) y * width + (int) x] = p.z;
    }
}

__device__ float edge_function(Point* a, Point* b, Point* c) {
    // printf("edge func: %f %f %f %f %f %f\n", a->x, a->y, b->x, b->y, c->x, c->y);
    return (c->x - a->x) * (b->y - a->y) - (c->y - a->y) * (b->x - a->x);
}
