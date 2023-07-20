#include "window.h"
#include <SDL_stdinc.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

Window::Window(const char* title, int width, int height) {
    this->window = SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                    width, height, SDL_WINDOW_SHOWN);
    this->surface = SDL_GetWindowSurface(this->window);
    this->width = width;
    this->height = height;
}

Window::~Window() {
    // SDL_DestroyRenderer(this->renderer);
    SDL_DestroyWindow(this->window);
}

void Window::clear(int r, int g, int b) {
    SDL_FillRect(this->surface, NULL, SDL_MapRGB(this->surface->format, r, g, b));
}

void Window::update() {
    SDL_UpdateWindowSurface(this->window);
}

void Window::poll_events() {
    clear(255, 255, 255);
    // draw_line({-100, -100, 255, 255, 255}, {100, 100, 255, 255, 255});
    draw_triangle({-100, -100, 255, 0, 0}, {0, 100, 0, 255, 0}, {100, -100, 0, 0, 255});

    put_pixel({0, 0, 255, 255, 255});

    update();

    SDL_Event event;
    bool running = true;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }
    }
}

void Window::put_pixel(Point p) {
    p = point_to_screen(p);

    // update single pixel in surface buffer
    auto surface = SDL_GetWindowSurface(this->window);
    Uint8* pixel = (Uint8*) surface->pixels;
    pixel += ((int) p.y * surface->pitch) + ((int) p.x * sizeof(Uint32));
    pixel[0] = p.r;
    pixel[1] = p.g;
    pixel[2] = p.b;
}

Point Window::point_to_screen(Point p) {
    // convert coordinates to screen
    p.x = this->width / 2.0f + p.x;
    p.y = this->height / 2.0f - p.y;

    return p;
}

float Window::edge_function(Point a, Point b, Point c) {
    // edge function
    // (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)

    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

void Window::draw_triangle(Point a, Point b, Point c) {

    float area = edge_function(a, b, c);

    for (int x = -this->width / 2; x < this->width / 2; x++) {
        for (int y = -this->height / 2; y < this->height / 2; y++) {
            Point p = {(float) x, (float) y, 255, 255, 255};

            float w0 = edge_function(b, c, p) / area;
            float w1 = edge_function(c, a, p) / area;
            float w2 = edge_function(a, b, p) / area;

            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                p.r = w0 * a.r + w1 * b.r + w2 * c.r;
                p.g = w0 * a.g + w1 * b.g + w2 * c.g;
                p.b = w0 * a.b + w1 * b.b + w2 * c.b;
                put_pixel(p);
            }
        }
    }
}

void Window::draw_line(Point p0, Point p1) {
    // bresenham line algorithm
    int dx = std::abs(p1.x - p0.x);
    int dy = -std::abs(p1.y - p0.y);
    int sx = (p0.x < p1.x) ? 1 : -1;
    int sy = (p0.y < p1.y) ? 1 : -1;
    int err = dx + dy;

    int e2;

    while (true) {
        put_pixel(p0);

        if (p0.x == p1.x && p0.y == p1.y) {
            break;
        }

        e2 = 2 * err;

        if (e2 >= dy) {
            if (p0.x == p1.x) break;
            err += dy;
            p0.x += sx;
        }

        if (e2 <= dx) {
            if (p0.y == p1.y) break;
            err += dx;
            p0.y += sy;
        }
    }
}
