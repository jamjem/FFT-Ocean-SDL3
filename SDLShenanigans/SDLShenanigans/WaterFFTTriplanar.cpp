#define SDL_MAIN_USE_CALLBACKS
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "mfg.hpp"

namespace
{
    /*
        https://jtessen.people.clemson.edu/reports/papers_files/coursenotes2004.pdf
        https://barthpaleologue.github.io/Blog/posts/ocean-simulation-webgpu/ the barth in question
        ===================================================================
        WHAT THIS SHIT DO
        ===================================================================
        a wave has:
             a direction 
             a frequency 
        slap those with a vector in frequency space
            k = wdirection * wfrequency

        for every one of those k vectors, the phillips spectrum tells us how
        much amplitude that wave gets

            P(k) = A * exp(-1/(k*L)^2) / k^4 * |k_hat . w_hat|^2

        the dot between k_hat and w_hat biases energy toward waves travelling
        with the wind and kills waves going sideways to the wind. raising it
        to a power tightens the cone.

        settings Barthélémy Paleologue used:
            A = 1.0            amplitude
            L = 1000 m         ocean tile size
            |w| = 31 m/s       wind speed
            resolution = 256   N in the NxN frequency grid

        that spits out a 256x256 texture that encodes the spectrum amplitude.
        put that into an sdl texture and draw it to the window. YIPPEEE
    */

    constexpr float    kPi          = 3.14159265359f;

	// the settings for the phillips spectrum.
    constexpr int      kFFTSize     = 256;     // N
    constexpr float    kPatchLength = 1000.0f; // L, in meters
    constexpr float    kPhillipsAmp = 1.0f;    // Amplitude
    constexpr float    kWindSpeed   = 31.0f;   // w in m/s
    constexpr float    kWindPower   = 2.0f;    // p on |k_hat . w_hat|^p (classic phillips = 2)
    constexpr float    kGravity     = 9.81f;   // g, needed by the max wavelength L_wave = V^2/g

    constexpr int      kWindowSize  = 768;

    // fft bins run 0..N-1 but wave vectors need to be centered on 0. anything
    // past N/2 wraps into negative land. then multiply by 2*pi/L to get the
    // physical k in radians per meter. returning a vec2 from the glorius mfg library so phillips can just dot/magnitude it directly
    static mfg::vec2 KVector(int x, int y, int n, float patchLength)
    {
        const int mx = (x <= (n / 2)) ? x : (x - n);
        const int my = (y <= (n / 2)) ? y : (y - n);
        const float scale = (2.0f * kPi) / patchLength;
        return mfg::vec2(static_cast<float>(mx) * scale,
                         static_cast<float>(my) * scale);
    }

    // phillips spectrum, step one for making the oceans heightmap
	//   P(k) = A * exp(-1/(kL)^2) / k^4 * |k_hat . w_hat|^p   (this is the formula from barth)
    //
    // the 1/k^4 term pumps energy into long waves, the exp(-1/(kL)^2) kills
    // anything bigger than the patch, and |k_hat . w_hat|^p is the wind
    // alignment factor that makes the ocean look wind-driven instead of
    // a uniform chop in every direction.
    static float Phillips(const mfg::vec2& k,
                          const mfg::vec2& windDir,
                          float windSpeed,
                          float windPower)
    {
        const float kLen = mfg::Magnitude(k);
        if (kLen < 1e-6f)
        {
            return 0.0f;
        }
        const float k2 = kLen * kLen;
        const float k4 = k2 * k2;

        // L_wave = V^2/g is the biggest wave the wind can whip up under deep
        // water dispersion. This is NOT kPatchLength (the tile size).
        const float L  = (windSpeed * windSpeed) / kGravity;
        const float L2 = L * L;

        // k_hat . w_hat, abs'd, raised to p. assumes windDir is already unit
        // length (we normalize it once up in main, so thats fine) what a stupid set of made up words
        const mfg::vec2 kHat = k / kLen;
        const float kDotW = mfg::Dot(kHat, windDir);
        const float alignment = std::pow(std::fabs(kDotW), windPower);

        const float expLarge = std::exp(-1.0f / (k2 * L2));
        return kPhillipsAmp * (expLarge / k4) * alignment;
    }

    // build a 256x256 grayscale texture. each pixel =
    // one bin in the frequency grid, value = phillips amplitude at that k.
    //
    // phillips has absurd dynamic range (the 1/k^4 term) so I cant just
    // dump P(k) into a byte and call it a day. ergo solutions:
    //   - take sqrt to compress the scale
    //   - then normalize against the max across the whole texture
    //   - throw a mild gamma on for looking cool
    // output is fftshifted: the k=0 bin lands at the center of the image so a wind-aligned butcheek shows up. 
    static void BuildPhillipsTexture(std::vector<uint8_t>& rgba,
                                     const mfg::vec2& wind,
                                     float windSpeed,
                                     float windPower)
    {
        const int N = kFFTSize;
        rgba.assign(N * N * 4, 0u);

        // first pass: dump raw P values into a float buffer and find the max
        std::vector<float> values(N * N, 0.0f);
        float maxP = 0.0f;
        for (int y = 0; y < N; ++y)
        {
            for (int x = 0; x < N; ++x)
            {
                const mfg::vec2 k = KVector(x, y, N, kPatchLength);
                const float p = Phillips(k, wind, windSpeed, windPower);
                values[y * N + x] = p;
                if (p > maxP) { maxP = p; }
            }
        }

        // safety valve to not divide by zero on a dead spectrum
        if (maxP <= 0.0f)
        {
            maxP = 1.0f;
        }

        // second pass: fftshift + sqrt/normalize/gamma into an 8-bit grayscale
        const float invMax = 1.0f / maxP;
        for (int y = 0; y < N; ++y)
        {
            for (int x = 0; x < N; ++x)
            {
                // fftshift: swap halves so k=(0,0) ends up in the middle
                const int sx = (x + N / 2) % N;
                const int sy = (y + N / 2) % N;
                const float p = values[sy * N + sx];

                // sqrt compresses the head of the curve, normalize brings it
                // into 0..1, gamma 0.65 gives it a bit more bite so faint
                // off-axis waves are still visible. clamp manually because idk I dont have clamp, prob the c++ version
                float v = std::sqrt(std::max(0.0f, p) * invMax);
                if (v < 0.0f) { v = 0.0f; }
                if (v > 1.0f) { v = 1.0f; }
                v = std::pow(v, 0.65f);

                const uint8_t byte = static_cast<uint8_t>(v * 255.0f + 0.5f);
                const int idx = (y * N + x) * 4;
                rgba[idx + 0] = byte;
                rgba[idx + 1] = byte;
                rgba[idx + 2] = byte;
                rgba[idx + 3] = 255u;
            }
        }
    }

    struct AppState
    {
        SDL_Window*   window   = nullptr;
        SDL_Renderer* renderer = nullptr;
        SDL_Texture*  spectrum = nullptr;
		std::vector<uint8_t> pixels; //might wanna get rid of this after we upload to the gpu, it small but liek
    };
}

SDL_AppResult SDL_AppInit(void** appstate, int /*argc*/, char** /*argv*/)
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    AppState* app = new AppState();
    *appstate = app;

    // ask for the window + renderer in one call cuz sdl is baller
    if (!SDL_CreateWindowAndRenderer("Let there be light",
                                     kWindowSize, kWindowSize, 0,
                                     &app->window, &app->renderer))
    {
        SDL_Log("SDL_CreateWindowAndRenderer failed: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // nearest filter so each spectrum bin stays a sharp square. linear blur
    // would hide the grid pattern which is the whole point of visualising it rn
    SDL_SetRenderLogicalPresentation(app->renderer, kFFTSize, kFFTSize,
                                     SDL_LOGICAL_PRESENTATION_LETTERBOX);

    // wind vector. magnitude = wind speed, direction in the xz plane.
    // slight off-axis tilt just so the result isnt pixel-perfectly axis
    // aligned, makes the butcheeks more obvious
    const mfg::vec2 windRaw(1.0f, 0.28f);
    const mfg::vec2 windDir = mfg::Normalize(windRaw);

    BuildPhillipsTexture(app->pixels, windDir, kWindSpeed, kWindPower);

    // upload the cpu-side bytes into a gpu-side texture. ABGR8888 in sdl3
    // matches the R,G,B,A byte order i wrote into the vector above. gonna be honest a little lost here
    app->spectrum = SDL_CreateTexture(app->renderer,
                                      SDL_PIXELFORMAT_ABGR8888,
                                      SDL_TEXTUREACCESS_STATIC,
                                      kFFTSize, kFFTSize);
    if (!app->spectrum)
    {
        SDL_Log("SDL_CreateTexture failed: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }
    SDL_SetTextureScaleMode(app->spectrum, SDL_SCALEMODE_NEAREST);
    SDL_UpdateTexture(app->spectrum, nullptr, app->pixels.data(),
                      kFFTSize * 4);

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate)
{
    AppState* app = static_cast<AppState*>(appstate);

    // clear to sorta dark grey so black bits of the spectrum still read as
    // data not gone
    SDL_SetRenderDrawColor(app->renderer, 12, 12, 18, 255);
    SDL_RenderClear(app->renderer);

    // logical presentation is already kFFTSize x kFFTSize so drawing the
    // texture at its natural size fills the logical surface
    SDL_FRect dst{ 0.0f, 0.0f,
                   static_cast<float>(kFFTSize),
                   static_cast<float>(kFFTSize) };
    SDL_RenderTexture(app->renderer, app->spectrum, nullptr, &dst);

    SDL_RenderPresent(app->renderer);
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event)
{
    AppState* app = static_cast<AppState*>(appstate);

    if (event->type == SDL_EVENT_QUIT ||
        event->type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
    {
        return SDL_APP_SUCCESS;
    }

    // S dumps the spectrum to a BMP 
    if (event->type == SDL_EVENT_KEY_DOWN &&
        event->key.key == SDLK_S &&
        app && !app->pixels.empty())
    {
        SDL_Surface* surf = SDL_CreateSurfaceFrom(kFFTSize, kFFTSize,
                                                  SDL_PIXELFORMAT_ABGR8888,
                                                  app->pixels.data(),
                                                  kFFTSize * 4);
        if (surf)
        {
            SDL_SaveBMP(surf, "phillips_spectrum.bmp");
            SDL_DestroySurface(surf);
            SDL_Log("saved phillips_spectrum.bmp");
        }
    }

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult /*result*/)
{
    AppState* app = static_cast<AppState*>(appstate);
    if (!app) { return; }

    if (app->spectrum) { SDL_DestroyTexture(app->spectrum); }
    if (app->renderer) { SDL_DestroyRenderer(app->renderer); }
    if (app->window)   { SDL_DestroyWindow(app->window); }
    delete app;
}
