#include "window.h"
#include <SDL_stdinc.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>

Window::Window(const char* title, int width, int height) {
    this->window = SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                    width, height, SDL_WINDOW_SHOWN);
    this->renderer = SDL_CreateRenderer(this->window, -1, SDL_RENDERER_ACCELERATED);
    this->surface = SDL_GetWindowSurface(this->window);
    this->width = width;
    this->height = height;
}

Window::~Window() {
    SDL_DestroyRenderer(this->renderer);
    SDL_DestroyWindow(this->window);
}

void Window::clear() {
    SDL_RenderClear(this->renderer);
}

void Window::set_color(int r, int g, int b) {
    SDL_SetRenderDrawColor(this->renderer, r, g, b, 255);
}

void Window::update() {
    SDL_UpdateWindowSurface(this->window);
    // SDL_RenderPresent(this->renderer);
}

void Window::poll_events() {
    set_color(0, 0, 0);
    clear();
    draw_line({-100, -100, 255, 255, 255}, {100, 100, 255, 255, 255});
    // draw_triangle({-100, -100, 255, 0, 0}, {0, 100, 0, 255, 0}, {100, -100, 0, 0, 255});

    put_pixel({0, 0, 255, 255, 255});

    update();

    SDL_Event event;
    bool running = true;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }

            // put_pixel({0, 0, 255, 255, 255, 255});
            // draw_line({-100, -100, 255, 255, 255, 255}, {100, 100, 255, 255, 255, 255});

        }
    }
}

void Window::put_pixel(Point p) {
    p = point_to_screen(p);
    // set_color(p.r, p.g, p.b);
    // SDL_RenderDrawPoint(this->renderer, p.x, p.y);

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

void Window::draw_triangle(Point a, Point b, Point c) {

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
