// Minimal SDL2 bindings for the visual particle demo.
pub const c = @cImport({
    @cDefine("SDL_MAIN_HANDLED", "1");
    @cInclude("SDL.h");
    @cInclude("SDL_render.h");
});
