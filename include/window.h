#include <SDL2/SDL.h>

class Window {
    public:
        Window(const char* title, int width, int height);
        ~Window();

        void clear();
        void update();
        void poll_events();
        void set_color(int r, int g, int b, int a);

        void put_pixel(int x, int y);

    private:
        SDL_Window* window;
        SDL_Renderer* renderer;
};
