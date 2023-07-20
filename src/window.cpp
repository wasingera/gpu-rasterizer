#include "window.h"
#include <cmath>

Window::Window(const char* title, int width, int height) {
    this->window = SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                    width, height, SDL_WINDOW_SHOWN);
    this->renderer = SDL_CreateRenderer(this->window, -1, SDL_RENDERER_ACCELERATED);
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

void Window::set_color(int r, int g, int b, int a) {
    SDL_SetRenderDrawColor(this->renderer, r, g, b, a);
}

void Window::update() {
    SDL_RenderPresent(this->renderer);
}

void Window::poll_events() {
    set_color(0, 0, 0, 255);
    clear();

    SDL_Event event;
    bool running = true;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }

            put_pixel({0, 0, 255, 255, 255, 255});

            update();
        }
    }
}

void Window::put_pixel(Point p) {
    // convert coordinates to screen
    p.x = this->width / 2.0f + p.x;
    p.y = this->height / 2.0f - p.y;

    set_color(p.r, p.g, p.b, p.a);
    SDL_RenderDrawPoint(this->renderer, p.x, p.y);
}

void Window::draw_line(Point p0, Point p1) {
    // bresenham line algorithm
    int dx = std::abs(p1.x - p0.x);
    int dy = std::abs(p1.y - p0.y);
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
