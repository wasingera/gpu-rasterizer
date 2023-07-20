#pragma once

#include <SDL2/SDL.h>

typedef struct {
    float x;
    float y;
    float r;
    float g;
    float b;
    float a;
} Point;

class Window {
    public:
        Window(const char* title, int width, int height);
        ~Window();

        void clear();
        void update();
        void poll_events();
        void set_color(int r, int g, int b, int a);

        void put_pixel(Point p);
        void draw_line(Point p0, Point p1);

    private:
        SDL_Window* window;
        SDL_Renderer* renderer;
        int width;
        int height;
};
