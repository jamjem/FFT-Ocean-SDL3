#define SDL_MAIN_USE_CALLBACKS
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <cstdint>
#include <cmath>
#include <random>
#include <utility>
#include <vector>
#include "mfg.hpp"

namespace
{
    /*
        https://jtessen.people.clemson.edu/reports/papers_files/coursenotes2004.pdf
        https://barthpaleologue.github.io/Blog/posts/ocean-simulation-webgpu/ the barth in question
        http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham4.html the wonders of Radix-4
        https://github.com/PD-2272835/MathsLib shrimp library
        ===================================================================
        WHAT THIS SHIT DO
        ===================================================================
        step 1 phillips spectrum:

            P(k) = A * exp(-1/(k*L)^2) / k^4 * |k_hat . w_hat|^p

        step 2 stops the determinism by multiplying in a complex gaussian:
            h0(k) = (1/sqrt(2)) * sqrt(P(k)) * (xi_r + i * xi_i)

        step 3 makes it wibbly wobbly and move. tessendorf's closed form moves h0 forward
        to any time t in one shot, without simulating anything in between.
         tessendorfs time evolution formula
            h(k, t) = h0(k) * exp(+i*w(k)*t)
                    + h0(-k)* * exp(-i*w(k)*t)

        with w(k) = sqrt(g * |k|) - the deep-water dispersion relation.

        step 4 pulls it back from frequency to real space with a 2d inverse
        fft, which is what finally gives me actual heights at actual points:

            h(x, t) = sum_k h~(k, t) * exp(+i k.x)

        the fft is separable meaning the 2d version is just
        a 1d fft on every row followed by a 1d fft on every column.
		stupid math terms and symbols are the bane of my existence in this project
        
        I made my own stockham radix-4 butterfly in a compute shader (fft_stockham.comp.glsl)
        instead of pulling in GLFFT because I wanted to write out the math cuz I still dont fully get it.
        one shader dispatch is one stage of the 1d fft, and each stage reads the result
        of the last one, so I need two textures to ping pong between
        (fftA/fftB): you can't read and write to the same texture in the same
        shader, so stage 1 reads fftA then writes fftB, stage 2 reads fftB then writes fftA
        and so on, hence ping pong. log4(N) stages along rows + log4(N) along
		columns = 2*log4(N) dispatches. for N=256 that's 8. Radix 2 which I started with would have been 16 dispatches
        so radix 4 runs significantly faster. Radix 2 is log2(n) stages

		the stockham compute shaders are both the same algorithm, 
        however fft_stockham.comp.glsl transforms the height spectrum 
        and fft_stockham2.comp.glsl transforms the slope spectrum, 
        which has 2 complex numbers per pixel instead of 1, 
        but the butterfly math is the same other than that

        step 5 is to compute the surface gradient alongside the height, so later
        I can build normals without finite differences. calculus bs:
        taking d/dx of h(x, t) = sum_k h~(k, t) exp(+i k.x) pulls an (i*k)
        out of every term. 
        The slope spectrum is just:

            slope_x(k, t) = i * k.x * h~(k, t)
            slope_z(k, t) = i * k.y * h~(k, t)

        Most of the calculus in this project goes over my head, and is probably 
        inefficient at the very least. I did my best to follow formulas that others
        used before me to do this

        The time_spectrum compute already has h(k, t) in a register
        so it writes the slope spectrum to a second RGBA32F output
        in the same dispatch (rg = x slope, ba = z slope). I then run the
        same separable ifft over it, using a vec4 version of the butterfly
        (fft_stockham2.comp.glsl) that does two complex multiplies per pixel
        one on .rg, one on .ba, so both components come out in one pass
        instead of two. the final texture drawn to screen shows the spatial
        gradient map: red channel is d h / d x, green channel is d h / d z.
        (for the devkits im stopping here and not making it 3d)

        gpu layout:
		  h0 is a RGBA32F texture - rg = h0(k), ba = h0(-k)*. 4 channel float texture, so each pixel can hold two complex numbers
          .rg = h0(k)   is the complex amplitude at frequency k
          .ba = h0(-k)* is the mirror frequency, pre-conjugated instead of flipping the sign every frame

          h_kt is RG32F, written by the time evolution compute every frame
          and fed into the first height fft pass as input.

          slope_kt is RGBA32F, written by the same compute pass, fed into
          the first slope fft pass.

          fftA / fftB are RG32F ping pong textures for the height ifft.
          slopeA / slopeB are RGBA32F ping pong textures for the slope
          ifft. both go through src -> A -> B -> A -> so on, and the final result
          lands in whichever is dst on the last pass (fftB/slopeB for N=256)
          the fragment shader currently draws the gradient
		  slopemap. 
          heightmap_view.frag.spv shows the greyscale heightmap instead.

        settings Barthélémy Paleologue used:
            A = 1.0           amplitude
            L = 1000 m        ocean tile size
           |w| = 31 m/s       wind speed
            p = 6             wind directionality
            N = 256           NxN frequency grid
        YIPPEEE
    */

    // simulation settings.
    constexpr int      kN          = 256;
    constexpr float    kPatch      = 1000.0f;  // Length, meters
    constexpr float    kAmp        = 1.0f;     // Amplitude
    constexpr float    kWindSpeed  = 31.0f;    // wind speed, m/s (|w|?)
    constexpr float    kWindPower  = 6.0f;     // classic phillips = 2
    constexpr float    kGravity    = 9.81f;
    constexpr int      kWindow     = 720;
    constexpr uint64_t kNoiseSeed  = 0xC05DC0FFEE5EEDEDull;

    // uniform block handed to initial_spectrum.comp.glsl. byte layout has to match std140 exactly
    struct PhillipsParams
    {
        int32_t N;            // 0
        float   patchLength;  // 4
        float   amplitude;    // 8
        float   windSpeed;    // 12
        float   windX;        // 16
        float   windY;        // 20
        float   windPower;    // 24
        float   gravity;      // 28
    };
    static_assert(sizeof(PhillipsParams) == 32, "PhillipsParams must match std140");

    // uniform block for time_spectrum.comp.glsl.
    struct TimeParams
    {
        int32_t N;            // 0
        float   patchLength;  // 4
        float   time;         // 8
        float   gravity;      // 12
    };
    static_assert(sizeof(TimeParams) == 16, "TimeParams must match std140");

    // uniform block for fft_stockham.comp.glsl.
    struct FFTParams
    {
        int32_t N;            // 0    texture size
        int32_t stage;        // 4    1..log2(N)
        int32_t horizontal;   // 8    1 = row pass, 0 = column pass
        float   sign;         // 12   +1 inverse, -1 forward
    };
    static_assert(sizeof(FFTParams) == 16, "FFTParams must match std140");

    // log2 at compile time is simpler than pulling <bit> in for one use
    constexpr int Log2(int n)
    {
        int r = 0;
        while (n > 1) { n >>= 1; ++r; }
        return r;
    }
    constexpr int kLog2N = Log2(kN);
    constexpr int kLog4N = kLog2N / 2;  // log base-4 of N; valid when N is a power of 4
    static_assert((1 << kLog2N) == kN,  "kN must be a power of two");
    static_assert(kLog2N % 2 == 0,      "kN must be a power of four for radix-4 fft");

    // everything the app owns. destruction order in SDL_AppQuit is the reverse of creation order here.
    struct App
    {
        SDL_Window*              window       = nullptr;
        SDL_GPUDevice*           device       = nullptr;
        SDL_GPUComputePipeline*  initCompute  = nullptr;  // phillips * noise -> h0
        SDL_GPUComputePipeline*  timeCompute  = nullptr;  // h0 -> h(k, t) + slope(k, t) every frame
        SDL_GPUComputePipeline*  fftCompute   = nullptr;  // one stockham stage, RG32F (height)
        SDL_GPUComputePipeline*  fftCompute2  = nullptr;  // one stockham stage, RGBA32F (slopes, 2 complex per pixel)
        SDL_GPUGraphicsPipeline* draw         = nullptr;  // fullscreen triangle -> gradient view
        SDL_GPUTexture*          noise        = nullptr;  // RG32F, the cpu gaussians
        SDL_GPUTexture*          h0           = nullptr;  // RGBA32F, (h0.rg, h0(-k)*.ba)
        SDL_GPUTexture*          h_kt         = nullptr;  // RG32F, h(k, t)
        SDL_GPUTexture*          slope_kt     = nullptr;  // RGBA32F, (i*kx*h.rg, i*kz*h.ba)
        SDL_GPUTexture*          fftA         = nullptr;  // RG32F, height ifft ping
        SDL_GPUTexture*          fftB         = nullptr;  // RG32F, height ifft pong
        SDL_GPUTexture*          slopeA       = nullptr;  // RGBA32F, slope ifft ping
        SDL_GPUTexture*          slopeB       = nullptr;  // RGBA32F, slope ifft pong
        SDL_GPUTexture*          heightmap    = nullptr;  // alias whichever of fftA/fftB as of the last pass
        SDL_GPUTexture*          slopemap     = nullptr;  // alias whichever of slopeA/slopeB as of the last pass
        SDL_GPUSampler*          sampler      = nullptr;  // nearest/repeat, used by every pass
    };
}

// ==========================================================================
// SDL_AppInit - create gpu device, make pipelines, upload noise, run h0 once
// ==========================================================================
SDL_AppResult SDL_AppInit(void** appstate, int, char**)
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("SDL_Init: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    App* app = new App();
    *appstate = app;

    // window + gpu device + hook them together. 
    app->window = SDL_CreateWindow("The universe sings to me", kWindow, kWindow, SDL_WINDOW_RESIZABLE);
    app->device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, nullptr);
    if (!app->window || !app->device ||
        !SDL_ClaimWindowForGPUDevice(app->device, app->window))
    {
        SDL_Log("gpu init: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // compute pipeline
    // just a shader object that runs on the gpu but doesn't draw anything,
    // it reads and writes textures/buffers and that's it. I use them
    // for the phillips spectrum, time evolution and the fft. the counts
    // that are passed into numSamplers, numRWStorageTex, and numUniformBufs tell sdl
    // how many of each resource type the shader expects to have bound;
    // they have to match what the glsl actually declares or pipeline explodes.
    auto LoadCompute = [&](const char* path,
                           Uint32 numSamplers,
                           Uint32 numRWStorageTex,
                           Uint32 numUniformBufs) -> SDL_GPUComputePipeline*
    {
        size_t sz   = 0;
        void*  code = SDL_LoadFile(path, &sz);
        if (!code) { SDL_Log("%s: %s", path, SDL_GetError()); return nullptr; }

        SDL_GPUComputePipelineCreateInfo ci{};
        ci.code = static_cast<const Uint8*>(code);
        ci.code_size = sz;
        ci.entrypoint = "main";
        ci.format = SDL_GPU_SHADERFORMAT_SPIRV;
        ci.num_samplers = numSamplers;
        ci.num_readwrite_storage_textures = numRWStorageTex;
        ci.num_uniform_buffers = numUniformBufs;
        ci.threadcount_x = 8;
        ci.threadcount_y = 8;
        ci.threadcount_z = 1;

        SDL_GPUComputePipeline* p = SDL_CreateGPUComputePipeline(app->device, &ci);
        SDL_free(code);
        return p;
    };

    // compute pipelines
    //
    // SDL_GPU has a fixed convention for compute shaders.
    // 0 is read only sampled textures
	// 1 is read/write storage textures
	// 2 is uniform buffers
	// 3 is read only storage buffers/textures
    // 
    // 
    // binding layouts:
    //   initCompute (initial_spectrum.comp.glsl)
    //     set=0.0  sampler2D u_noise
    //     set=1.0  image2D   u_h0(rgba32f)
    //     set=2.0  uniform   PhillipsParams
    //
    //   timeCompute (time_spectrum.comp.glsl)
    //     set=0.0  sampler2D u_h0
    //     set=1.0  image2D   u_h(rg32f,   h(k, t))
    //     set=1.1  image2D   u_slope(rgba32f, i*k*h(k, t))
    //     set=2.0  uniform   TimeParams
    //
    //   fftCompute / fftCompute2 (fft_stockham.comp.glsl, fft_stockham2.comp.glsl)
    //     set=0.0  sampler2D u_src
    //     set=1.0  image2D   u_dst  (rg32f/rgba32f)
    //     set=2.0  uniform   FFTParams
    app->initCompute = LoadCompute("Shaders/initial_spectrum.comp.spv", 1, 1, 1);
    app->timeCompute = LoadCompute("Shaders/time_spectrum.comp.spv",    1, 2, 1);
    app->fftCompute  = LoadCompute("Shaders/fft_stockham.comp.spv",     1, 1, 1);
    app->fftCompute2 = LoadCompute("Shaders/fft_stockham2.comp.spv",    1, 1, 1);
    if (!app->initCompute || !app->timeCompute || !app->fftCompute || !app->fftCompute2)
    {
        SDL_Log("compute pipeline creation failed: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // tiny loader for graphics shaders.
    auto LoadShader = [&](const char* path,
                          SDL_GPUShaderStage stage,
                          Uint32 samplers,
                          Uint32 uniformBuffers) -> SDL_GPUShader*
    {
        size_t sz = 0;
        void*  code = SDL_LoadFile(path, &sz);
        if (!code) { SDL_Log("%s: %s", path, SDL_GetError()); return nullptr; }

        SDL_GPUShaderCreateInfo ci{};
        ci.code = static_cast<const Uint8*>(code);
        ci.code_size = sz;
        ci.entrypoint = "main";
        ci.format = SDL_GPU_SHADERFORMAT_SPIRV;
        ci.stage = stage;
        ci.num_samplers = samplers;
        ci.num_uniform_buffers = uniformBuffers;

        SDL_GPUShader* s = SDL_CreateGPUShader(app->device, &ci);
        SDL_free(code);
        return s;
    };

    // fullscreen triangle
    // I draw one oversized triangle with just 3 vertices generated
    // inside the vertex shader from gl_VertexIndex. 
    // it covers the whole screen and has no vertex buffer
    // fragment shader then does all the work, here gradient_view samples the final slope ifft texture and paints
    // (d h / d x, d h / d z) as red/green.
    SDL_GPUShader* vs = LoadShader("Shaders/fullscreen.vert.spv",    SDL_GPU_SHADERSTAGE_VERTEX,   0, 0);
    SDL_GPUShader* fs = LoadShader("Shaders/gradient_view.frag.spv", SDL_GPU_SHADERSTAGE_FRAGMENT, 1, 0);
    if (!vs || !fs) { SDL_Log("shader load: %s", SDL_GetError()); return SDL_APP_FAILURE; }

    SDL_GPUColorTargetDescription colorTarget{};
    colorTarget.format = SDL_GetGPUSwapchainTextureFormat(app->device, app->window);

    // pipeline state for the fullscreen pass. triangle list primitive
    // type, no vertex input since the shader makes its own verts. 
    // front-face order and cull_mode don't really matter here since the triangle covers everything
    SDL_GPUGraphicsPipelineCreateInfo gpci{};
    gpci.vertex_shader   = vs;
    gpci.fragment_shader = fs;
    gpci.primitive_type  = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    gpci.rasterizer_state.fill_mode  = SDL_GPU_FILLMODE_FILL;
    gpci.rasterizer_state.cull_mode  = SDL_GPU_CULLMODE_NONE;
    gpci.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
    gpci.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
    gpci.target_info.num_color_targets         = 1;
    gpci.target_info.color_target_descriptions = &colorTarget;

    app->draw = SDL_CreateGPUGraphicsPipeline(app->device, &gpci);
    // pipeline keeps its own reference to the shaders, i can let mine go.
    SDL_ReleaseGPUShader(app->device, vs);
    SDL_ReleaseGPUShader(app->device, fs);
    if (!app->draw) { SDL_Log("graphics pipeline: %s", SDL_GetError()); return SDL_APP_FAILURE; }

    // textures. every texture in SDL_GPU has a "usage" flag that tells
    // the driver how i'll use it - SAMPLER means the shader will sample
    // it like a normal texture (read-only, filtered), COMPUTE_STORAGE_WRITE
    // means a compute shader will write raw pixels into it. most of my
    // textures need both because they're written by one compute pass and
    // then sampled by the next.
    //
    // noise: the cpu-generated gaussians, read-only (SAMPLER only).
    // h0: 4-channel to hold both h0(k) and h0(-k)*. written by the init
    //     compute once, sampled by the time compute every frame.
    // h_kt: 2-channel complex spectrum h(k, t). written by the time
    //     compute every frame, feeds the first height ifft pass.
    SDL_GPUTextureCreateInfo tci{};
    tci.type = SDL_GPU_TEXTURETYPE_2D;
    tci.width = kN;
    tci.height = kN;
    tci.layer_count_or_depth = 1;
    tci.num_levels = 1;

    tci.format = SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT;
    tci.usage  = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    app->noise = SDL_CreateGPUTexture(app->device, &tci);

    tci.format = SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT;
    tci.usage  = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
    app->h0    = SDL_CreateGPUTexture(app->device, &tci);

    tci.format = SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT;
    tci.usage  = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
    app->h_kt  = SDL_CreateGPUTexture(app->device, &tci);

    // slope spectrum: the 2 complex numbers packed into RGBA32F
    //   rg = i * kx * h(k, t),  ba = i * kz * h(k, t)
    tci.format    = SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT;
    tci.usage     = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
    app->slope_kt = SDL_CreateGPUTexture(app->device, &tci);

    // height ifft pingpong pair. Each fft stage reads the result of the
    // previous one, so i can't read+write the same texture in one shader
    // A is src / B is dst, then B is src / A is dst,
    // then A / B again, so on so forth. 2*log2(N) stages in total per frame.
    tci.format = SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT;
    tci.usage  = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
    app->fftA  = SDL_CreateGPUTexture(app->device, &tci);
    app->fftB  = SDL_CreateGPUTexture(app->device, &tci);

    // slope ifft ping pong pair. same ping pong scheme as above, but
    // rgba32f so both complex numbers (x slope and z slope) travel
    // through the fft together.
    tci.format  = SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT;
    tci.usage   = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
    app->slopeA = SDL_CreateGPUTexture(app->device, &tci);
    app->slopeB = SDL_CreateGPUTexture(app->device, &tci);

    // which of the ping pong textures ends up holding the final pass
    // depends on the parity of 2*log4(N). simulate the swaps now so the
    // fragment shader can bind the right one without a runtime check.
    // height and slope do the same number of passes so they pick the
    // same parity, but keeping them separate makes the intent clearer.
    {
        SDL_GPUTexture* dst   = app->fftA;
        SDL_GPUTexture* other = app->fftB;
        for (int p = 1; p < 2 * kLog4N; ++p) { std::swap(dst, other); }
        app->heightmap = dst;
    }
    {
        SDL_GPUTexture* dst   = app->slopeA;
        SDL_GPUTexture* other = app->slopeB;
        for (int p = 1; p < 2 * kLog4N; ++p) { std::swap(dst, other); }
        app->slopemap = dst;
    }

    // a sampler describes how a shader reads a texture
    // nearest vs linear filtering, repeat vs clamp, that kinda shtuff
    // I use one sampler for everything and it's nearest/repeat.
    // nearest keeps each pixel as sharp square when the window is bigger than N
    // the compute passes all use texelFetch, which ignores
    // filtering anyway, so they don't care what I pick here.
    SDL_GPUSamplerCreateInfo sci{};
    sci.min_filter = SDL_GPU_FILTER_NEAREST;
    sci.mag_filter = SDL_GPU_FILTER_NEAREST;
    sci.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    sci.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sci.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sci.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    app->sampler = SDL_CreateGPUSampler(app->device, &sci);

    if (!app->noise || !app->h0 || !app->h_kt || !app->slope_kt
        || !app->fftA || !app->fftB || !app->slopeA || !app->slopeB
        || !app->sampler)
    {
        SDL_Log("texture/sampler creation: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // fill then upload the gaussian noise
    std::vector<float> noisePixels(static_cast<size_t>(kN) * kN * 2);
    {
        std::mt19937_64 rng(kNoiseSeed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float& f : noisePixels) { f = dist(rng); }
    }

    SDL_GPUTransferBufferCreateInfo tbci{};
    tbci.size  = static_cast<Uint32>(noisePixels.size() * sizeof(float));
    tbci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    SDL_GPUTransferBuffer* tb = SDL_CreateGPUTransferBuffer(app->device, &tbci);

    void* mapped = SDL_MapGPUTransferBuffer(app->device, tb, false);
    SDL_memcpy(mapped, noisePixels.data(), tbci.size);
    SDL_UnmapGPUTransferBuffer(app->device, tb);

    SDL_GPUCommandBuffer* uploadCmd = SDL_AcquireGPUCommandBuffer(app->device);
    SDL_GPUCopyPass*      copyPass  = SDL_BeginGPUCopyPass(uploadCmd);
    SDL_GPUTextureTransferInfo src{};
    src.transfer_buffer = tb;
    src.pixels_per_row  = kN;
    src.rows_per_layer  = kN;
    SDL_GPUTextureRegion region{};
    region.texture = app->noise;
    region.w = kN; region.h = kN; region.d = 1;
    SDL_UploadToGPUTexture(copyPass, &src, &region, false);
    SDL_EndGPUCopyPass(copyPass);
    SDL_SubmitGPUCommandBuffer(uploadCmd);
    SDL_ReleaseGPUTransferBuffer(app->device, tb);

    // dispatch the init compute exactly once. A dispatch kicks off a
    // compute shader with some number of thread groups.
    // here I run (N/8) x (N/8) groups of 8x8 threads, so one gpu thread per texel.
    // fills h0 with h0(k) in rg + h0(-k)* in ba. never runs again after
    // this; the per-frame time compute reads it and produces h(k, t).
    SDL_GPUCommandBuffer* computeCmd = SDL_AcquireGPUCommandBuffer(app->device);

    SDL_GPUStorageTextureReadWriteBinding rw{};
    rw.texture = app->h0;
    SDL_GPUComputePass* cpass = SDL_BeginGPUComputePass(computeCmd, &rw, 1, nullptr, 0);

    SDL_BindGPUComputePipeline(cpass, app->initCompute);

    SDL_GPUTextureSamplerBinding noiseBinding{};
    noiseBinding.texture = app->noise;
    noiseBinding.sampler = app->sampler;
    SDL_BindGPUComputeSamplers(cpass, 0, &noiseBinding, 1);

    // slight off axis tilt so the wind alignment butterfly isn't pixel locked
    // to a row/column. normalising in the shrimp library is cheap and happens
    // once, the shader assumes it's already unit length.
    const mfg::vec2 windDir = mfg::Normalize(mfg::vec2(1.0f, 0.28f));
    PhillipsParams pp{};
    pp.N = kN;
    pp.patchLength = kPatch;
    pp.amplitude = kAmp;
    pp.windSpeed = kWindSpeed;
    pp.windX = windDir.x();
    pp.windY = windDir.y();
    pp.windPower = kWindPower;
    pp.gravity = kGravity;
    SDL_PushGPUComputeUniformData(computeCmd, 0, &pp, sizeof(pp));

    SDL_DispatchGPUCompute(cpass, (kN + 7) / 8, (kN + 7) / 8, 1);
    SDL_EndGPUComputePass(cpass);
    SDL_SubmitGPUCommandBuffer(computeCmd);

    return SDL_APP_CONTINUE;
}

// =============================================================================
// SDL_AppIterate - dispatch the time evolution, ifft it, draw the gradient map
// =============================================================================
SDL_AppResult SDL_AppIterate(void* appstate)
{
    App* app = static_cast<App*>(appstate);

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(app->device);

    // compute pass:
    // advance the spectrum to the current time.
    // SDL_GetTicks is in ms, milliseconds since SDL init. seconds is what the formula expects. so like, multiply.
    const float t = static_cast<float>(SDL_GetTicks()) * 0.001f;

    // time evolution: 
    // writes h(k, t) to h_kt AND slope(k, t) to slope_kt.
    // two rw storage textures are bound simultaneously, hence the array.
    SDL_GPUStorageTextureReadWriteBinding timeRW[2]{};
    timeRW[0].texture = app->h_kt;
    timeRW[1].texture = app->slope_kt;
    SDL_GPUComputePass* cpass = SDL_BeginGPUComputePass(cmd, timeRW, 2, nullptr, 0);
    SDL_BindGPUComputePipeline(cpass, app->timeCompute);

    SDL_GPUTextureSamplerBinding h0Binding{};
    h0Binding.texture = app->h0;
    h0Binding.sampler = app->sampler;
    SDL_BindGPUComputeSamplers(cpass, 0, &h0Binding, 1);

    TimeParams tp{};
    tp.N = kN;
    tp.patchLength = kPatch;
    tp.time = t;
    tp.gravity = kGravity;
    SDL_PushGPUComputeUniformData(cmd, 0, &tp, sizeof(tp));

    SDL_DispatchGPUCompute(cpass, (kN + 7) / 8, (kN + 7) / 8, 1);
    SDL_EndGPUComputePass(cpass);

    // -------------------------------------------------------------------
    // ifft passes: h(k, t) -> h(x, t)  and  slope(k, t) -> (dh/dx, dh/dz)
    //
    // separable 2d ifft = log4(N) horizontal stages (1d ifft per row) +
    // log4(N) vertical stages (1d ifft per column), pingponging between
    // two textures. the very first pass reads from the spectrum texture
    // (h_kt or slope_kt) instead of one of the ping-pong pair; after
    // that i just alternate src/dst.
    //
    // total passes: 2 * log4(N) per transform. for N=256 that's 8
    // compute passes per ifft, so 16 per frame across both transforms
    // (half my and evil old radix-2 count).
    // each is its own SDL compute pass, which matters because SDL then
    // knows to wait for the previous write to finish before the next read starts
    // without that, stage 2 would race stage 1 and you'd read garbage.
	// love you sdl <3
    // -------------------------------------------------------------------
    // dont forget only write the 2*log4(N) pass loop once and reuse it
    // for the height (rg32f) and slope (rgba32f) transforms.
    auto RunIFFT = [&](SDL_GPUTexture* src0,
                       SDL_GPUTexture* pingA,
                       SDL_GPUTexture* pingB,
                       SDL_GPUComputePipeline* pipeline)
    {
        SDL_GPUTexture* src = src0;
        SDL_GPUTexture* dst = pingA;
        for (int passIdx = 1; passIdx <= 2 * kLog4N; ++passIdx)
        {
            SDL_GPUStorageTextureReadWriteBinding rwTex{};
            rwTex.texture = dst;
            SDL_GPUComputePass* fp = SDL_BeginGPUComputePass(cmd, &rwTex, 1, nullptr, 0);
            SDL_BindGPUComputePipeline(fp, pipeline);

            SDL_GPUTextureSamplerBinding sb{};
            sb.texture = src;
            sb.sampler = app->sampler;
            SDL_BindGPUComputeSamplers(fp, 0, &sb, 1);

            const bool isHorizontal = (passIdx <= kLog4N);
            FFTParams fparams{};
            fparams.N          = kN;
            fparams.stage      = isHorizontal ? passIdx : (passIdx - kLog4N);
            fparams.horizontal = isHorizontal ? 1 : 0;
            fparams.sign       = +1.0f; // inverse dft
            SDL_PushGPUComputeUniformData(cmd, 0, &fparams, sizeof(fparams));

            SDL_DispatchGPUCompute(fp, (kN + 7) / 8, (kN + 7) / 8, 1);
            SDL_EndGPUComputePass(fp);

            // first pass reads from src0; after that it ping pongs between A and B.
            if (passIdx == 1) { src = pingA; dst = pingB; }
            else              { std::swap(src, dst); }
        }
    };

    RunIFFT(app->h_kt,     app->fftA,   app->fftB,   app->fftCompute);
    RunIFFT(app->slope_kt, app->slopeA, app->slopeB, app->fftCompute2);

    // ---- draw the gradient map as a fullscreen triangle (render pass) -------
    // the swapchain texture is the image I actually paint into for
    // this frame. SDL owns a small rotating pool of them, one of which
    // ends up on screen when I submit. it can come back null if the
    // window is minimised or mid-resize; if so I submit whatever's
    // already queued (so the FFT work isn't wasted) and try again next
    // tick.
    SDL_GPUTexture* swap = nullptr;
    Uint32 w = 0, h = 0;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd, app->window, &swap, &w, &h) || !swap)
    {
        SDL_SubmitGPUCommandBuffer(cmd);
        return SDL_APP_CONTINUE;
    }

    SDL_GPUColorTargetInfo ct{};
    ct.texture     = swap;
    ct.clear_color = { 0.0f, 0.0f, 0.0f, 1.0f };
    ct.load_op     = SDL_GPU_LOADOP_CLEAR;
    ct.store_op    = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &ct, 1, nullptr);
    SDL_BindGPUGraphicsPipeline(pass, app->draw);

    // fragment shader samples the gradient map at set 2 binding 0.
    // swap to app->heightmap if you rebind heightmap_view.frag.spv.
    SDL_GPUTextureSamplerBinding fragBinding{};
    fragBinding.texture = app->slopemap;
    fragBinding.sampler = app->sampler;
    SDL_BindGPUFragmentSamplers(pass, 0, &fragBinding, 1);

    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
    SDL_SubmitGPUCommandBuffer(cmd);
    return SDL_APP_CONTINUE;
}

// ==========================================================================
// SDL_AppEvent
// ==========================================================================
SDL_AppResult SDL_AppEvent(void* /*appstate*/, SDL_Event* event)
{
    if (event->type == SDL_EVENT_QUIT ||
        event->type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
    {
        return SDL_APP_SUCCESS;
    }
    return SDL_APP_CONTINUE;
}

// ==========================================================================
// SDL_AppQuit BURN IT ALL DOWN (In reverse order of its creation)
// ==========================================================================
void SDL_AppQuit(void* appstate, SDL_AppResult /*result*/)
{
    App* app = static_cast<App*>(appstate);
    if (!app) { return; }

    if (app->device)
    {
        if (app->draw)        { SDL_ReleaseGPUGraphicsPipeline(app->device, app->draw); }
        if (app->fftCompute2) { SDL_ReleaseGPUComputePipeline(app->device, app->fftCompute2); }
        if (app->fftCompute)  { SDL_ReleaseGPUComputePipeline(app->device, app->fftCompute); }
        if (app->timeCompute) { SDL_ReleaseGPUComputePipeline(app->device, app->timeCompute); }
        if (app->initCompute) { SDL_ReleaseGPUComputePipeline(app->device, app->initCompute); }
        if (app->sampler)     { SDL_ReleaseGPUSampler(app->device, app->sampler); }
        if (app->slopeB)      { SDL_ReleaseGPUTexture(app->device, app->slopeB); }
        if (app->slopeA)      { SDL_ReleaseGPUTexture(app->device, app->slopeA); }
        if (app->fftB)        { SDL_ReleaseGPUTexture(app->device, app->fftB); }
        if (app->fftA)        { SDL_ReleaseGPUTexture(app->device, app->fftA); }
        if (app->slope_kt)    { SDL_ReleaseGPUTexture(app->device, app->slope_kt); }
        if (app->h_kt)        { SDL_ReleaseGPUTexture(app->device, app->h_kt); }
        if (app->h0)          { SDL_ReleaseGPUTexture(app->device, app->h0); }
        if (app->noise)       { SDL_ReleaseGPUTexture(app->device, app->noise); }
        if (app->window)      { SDL_ReleaseWindowFromGPUDevice(app->device, app->window); }
        SDL_DestroyGPUDevice(app->device);
    }
    if (app->window) { SDL_DestroyWindow(app->window); }
    delete app;
}
