#pragma once

#include <SDL2/SDL.h>
#include <cuda_runtime.h>

typedef struct {
    float x;
    float y;
    float r;
    float g;
    float b;
} Point;


__device__ float edge_function(Point* a, Point* b, Point* c);

__global__ void draw_triangle_kernel(Point* a, Point* b, Point* c, void* pixels, int pitch, int width, int height);
// __global__ void draw_triangle_kernel(Point a, Point b, Point c) {

class Window {
    public:
        Window(const char* title, int width, int height);
        ~Window();

        void clear(int r, int g, int b);
        void update();
        void poll_events();
        void set_color(int r, int g, int b);

        void put_pixel(Point p);
        void draw_line(Point p0, Point p1);

        float edge_function(Point a, Point b, Point c);
        void draw_triangle(Point a, Point b, Point c);

        Point point_to_screen(Point p);

    private:
        SDL_Window* window;
        SDL_Surface* surface;
        int width;
        int height;
};
