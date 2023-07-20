#include "window.h"

__device__ float edge_function(Point a, Point b, Point c);

__global__ void draw_triangle(Point a, Point b, Point c) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    Point p = {(float) x, (float) y, 255, 255, 255};

    float area = edge_function(a, b, c);
    float w0 = edge_function(b, c, p) / area;
    float w1 = edge_function(c, a, p) / area;
    float w2 = edge_function(a, b, p) / area;

    // for (int x = -this->width / 2; x < this->width / 2; x++) {
    //     for (int y = -this->height / 2; y < this->height / 2; y++) {
    //         Point p = {(float) x, (float) y, 255, 255, 255};
    //
    //         float w0 = edge_function(b, c, p) / area;
    //         float w1 = edge_function(c, a, p) / area;
    //         float w2 = edge_function(a, b, p) / area;
    //
    //         if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
    //             p.r = w0 * a.r + w1 * b.r + w2 * c.r;
    //             p.g = w0 * a.g + w1 * b.g + w2 * c.g;
    //             p.b = w0 * a.b + w1 * b.b + w2 * c.b;
    //             put_pixel(p);
    //         }
    //     }
    // }
}

__device__ float edge_function(Point a, Point b, Point c) {
    // edge function
    // (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)

    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}
