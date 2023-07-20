#include "window.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

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

void Window::set_color(int r, int g, int b) {
    SDL_SetRenderDrawColor(this->renderer, r, g, b, 255);
}

void Window::update() {
    SDL_RenderPresent(this->renderer);
}

void Window::poll_events() {
    set_color(0, 0, 0);
    clear();
    draw_triangle({-100, -100, 255, 0, 0}, {0, 100, 0, 255, 0}, {100, -100, 0, 0, 255});

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
    set_color(p.r, p.g, p.b);
    SDL_RenderDrawPoint(this->renderer, p.x, p.y);
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
    // point to screen
    a = point_to_screen(a);
    b = point_to_screen(b);
    c = point_to_screen(c);

    // float min_x = std::min(a.x, std::min(b.x, c.x));
    // float max_x = std::max(a.x, std::max(b.x, c.x));
    //
    // float min_y = std::min(a.y, std::min(b.y, c.y));
    // float max_y = std::max(a.y, std::max(b.y, c.y));

    float area = edge_function(a, b, c);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
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
        put_pixel(point_to_screen(p0));

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
