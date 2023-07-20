// sdl2 code to create a window with an event loop

#include <iostream>
#include <SDL2/SDL.h>
#include "window.h"

int main(int argc, char* argv[]) {
    SDL_Init(SDL_INIT_VIDEO);

    Window* window = new Window("Rasterizer", 640, 480);

    window->poll_events();

    delete window;

    SDL_Quit();

    return 0;
}
