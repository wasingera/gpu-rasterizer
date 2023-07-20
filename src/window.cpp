#include "window.h"

Window::Window(const char* title, int width, int height) {
    this->window = SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                    width, height, SDL_WINDOW_SHOWN);
    this->renderer = SDL_CreateRenderer(this->window, -1, SDL_RENDERER_ACCELERATED);
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

            for (int i = 0; i < 340; i++) {
                for (int j = 0; j < 280; j++) {
                    put_pixel(i, j);
                }
            }

            update();
        }
    }
}

void Window::put_pixel(int x, int y) {
    set_color(255, 255, 255, 255);
    SDL_RenderDrawPoint(this->renderer, x, y);
}
