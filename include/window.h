#pragma once

#include <SDL2/SDL.h>

typedef struct {
    float x;
    float y;
    float r;
    float g;
    float b;
} Point;

class Window {
    public:
        Window(const char* title, int width, int height);
        ~Window();

        void clear();
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
        SDL_Renderer* renderer;
        int width;
        int height;
};
