#define SDL_MAIN_USE_CALLBACKS
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <cstdint>
#include <random>
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
        step 1 is the phillips spectrum:

            P(k) = A * exp(-1/(k*L)^2) / k^4 * |k_hat . w_hat|^p

            the dot between k_hat and w_hat biases energy toward waves
        travelling with the wind, raising it to p tightens the cone.

            but P(k) is deterministic. IFFT it and every patch of ocean
        looks identical, stamped out like a cookie cutter. real water
        has randomness at every wavevector. step 2 (tessendorf) fixes
        that by multiplying by a complex gaussian:

            h0(k) = (1/sqrt(2)) * sqrt(P(k)) * (xi_r + i * xi_i)
            
        step 2 gpu and gaussian noise added:

            everything that was CPU in step 1 has now moved to the gpu:
        cpu generates the gaussian noise once (box-muller via
        std::normal_distribution) and uploads it as an RG32F texture
        one compute dispatch evaluates P(k), reads the noise,
        multiplies them together, and writes h0(k) into an RG32F
        storage texture
        a fullscreen triangle samples h0 and draws it to the window
        with a fftshift so k=0 sits in the middle

        settings Barthélémy Paleologue used:
            A = 1.0            amplitude
            L = 1000 m         ocean tile size
            |w| = 31 m/s       wind speed
            p = 6              wind directionality (his showy screenshot)
            N = 256            N in the NxN frequency grid
        YIPPEEE
    */

  // ==========================================================================
  // Simulation Shtuff
  // ==========================================================================
	constexpr int      kN = 256;              // N in the NxN frequency grid
    constexpr float    kPatch      = 1000.0f;// L, meters
    constexpr float    kAmp        = 1.0f;  // Amplitude
    constexpr float    kWindSpeed  = 31.0f;// |w|, m/s
    constexpr float    kWindPower  = 6.0f;// p on |k_hat . w_hat|
    constexpr float    kGravity    = 9.81f;
    constexpr int      kWindow     = 768;
    constexpr uint64_t kNoiseSeed  = 0xC05DC0FFEE5EEDEDull;

    //  uniform block handed to initial_spectrum.comp.glsl. byte layout has
    // to match std140 exactly, int and float align to 4, vec2 aligns to 8,
    // and the block ends up rounded to a multiple of 16.
    struct Params
    {
        int32_t N;            // 0
        float   patchLength;  // 4
        float   amplitude;    // 8
        float   windSpeed;    // 12
        float   windX;        // 16  start of vec2 u_windDir (aligns to 8)
        float   windY;        // 20
        float   windPower;    // 24
        float   gravity;      // 28
    };
    static_assert(sizeof(Params) == 32, "Params must match std140");

    //destroying everything at the end is this in reverse
    struct App
    {
        SDL_Window*              window = nullptr;
        SDL_GPUDevice*           device = nullptr;
        SDL_GPUComputePipeline*  compute = nullptr;  // phillips * noise = h0
        SDL_GPUGraphicsPipeline* draw = nullptr;     // fullscreen tringle + view frag
        SDL_GPUTexture*          noise = nullptr;    // RG32F, the cpu gaussians
        SDL_GPUTexture*          h0 = nullptr;       // RG32F, h0(k)
        SDL_GPUSampler*          sampler = nullptr;  // nearest/repeat for both
    };
}

// =================================================================================
// SDL_AppInit - create gpu device, the pipelines, upload noise, run compute shaders
// =================================================================================
SDL_AppResult SDL_AppInit(void** appstate, int /*argc*/, char** /*argv*/)
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("SDL_Init: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    App* app = new App();
    *appstate = app;

    // window + gpu device + hook them together
    app->window = SDL_CreateWindow("TOO MUCH LIGHT", kWindow, kWindow, SDL_WINDOW_RESIZABLE);
    app->device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, nullptr);
    if (!app->window || !app->device ||
        !SDL_ClaimWindowForGPUDevice(app->device, app->window))
    {
        SDL_Log("gpu init: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // compute pipeline
    // initial_spectrum.comp.glsl has these bindings
    //     set=0 binding=0  sampler2D u_noise   (read the cpu gaussians)
    //     set=1 binding=0  image2D   u_h0      (write the complex spectrum)
    //     set=2 binding=0  uniform   Params    (push constants in practice)
    {
        size_t sz = 0;
        void*  code = SDL_LoadFile("Shaders/initial_spectrum.comp.spv", &sz);
        if (!code) { SDL_Log("initial_spectrum.comp.spv: %s", SDL_GetError()); return SDL_APP_FAILURE; }

        SDL_GPUComputePipelineCreateInfo ci{};
        ci.code = static_cast<const Uint8*>(code);
        ci.code_size = sz;
        ci.entrypoint = "main";
        ci.format = SDL_GPU_SHADERFORMAT_SPIRV;
        ci.num_samplers = 1;
        ci.num_readwrite_storage_textures = 1;
        ci.num_uniform_buffers  = 1;
        ci.threadcount_x = 8;
        ci.threadcount_y = 8;
        ci.threadcount_z = 1;
        app->compute = SDL_CreateGPUComputePipeline(app->device, &ci);
        SDL_free(code);
        if (!app->compute) { SDL_Log("compute pipeline: %s", SDL_GetError()); return SDL_APP_FAILURE; }
    }

    // graphics pipeline
    // vertex shader takes nothing, fragment samples h0 at set=2 binding=0
    auto LoadShader = [&](const char* path, SDL_GPUShaderStage stage, Uint32 samplers) -> SDL_GPUShader*
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

        SDL_GPUShader* s = SDL_CreateGPUShader(app->device, &ci);
        SDL_free(code);
        return s;
    };

    SDL_GPUShader* vs = LoadShader("Shaders/fullscreen.vert.spv",    SDL_GPU_SHADERSTAGE_VERTEX,   0);
    SDL_GPUShader* fs = LoadShader("Shaders/spectrum_view.frag.spv", SDL_GPU_SHADERSTAGE_FRAGMENT, 1);
    if (!vs || !fs) { return SDL_APP_FAILURE; }

    SDL_GPUColorTargetDescription colorTarget{};
    colorTarget.format = SDL_GetGPUSwapchainTextureFormat(app->device, app->window);

    SDL_GPUGraphicsPipelineCreateInfo gpci{};
    gpci.vertex_shader = vs;
    gpci.fragment_shader = fs;
    gpci.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    gpci.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    gpci.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
    gpci.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
    gpci.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
    gpci.target_info.num_color_targets = 1;
    gpci.target_info.color_target_descriptions = &colorTarget;

    app->draw = SDL_CreateGPUGraphicsPipeline(app->device, &gpci);
    // pipeline keeps its own reference to the shaders
    SDL_ReleaseGPUShader(app->device, vs);
    SDL_ReleaseGPUShader(app->device, fs);
    if (!app->draw) { SDL_Log("graphics pipeline: %s", SDL_GetError()); return SDL_APP_FAILURE; }

    // noise + h0 textures
    // both are NxN 2-channel floats. noise is read-only (SAMPLER), h0 is
    // both written by compute (COMPUTE_STORAGE_WRITE) and sampled by the
    // fragment shader (SAMPLER).
    SDL_GPUTextureCreateInfo tci{};
    tci.type = SDL_GPU_TEXTURETYPE_2D;
    tci.format = SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT;
    tci.width = kN;
    tci.height = kN;
    tci.layer_count_or_depth = 1;
    tci.num_levels = 1;

    tci.usage  = SDL_GPU_TEXTUREUSAGE_SAMPLER;
    app->noise = SDL_CreateGPUTexture(app->device, &tci);

    tci.usage  = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
    app->h0    = SDL_CreateGPUTexture(app->device, &tci);

    // nearest filter so each spectrum bin stays a sharp square when the
    // window is larger than N. linear would blur the grid pattern which
    // is exactly what we want to see while debugging.
    SDL_GPUSamplerCreateInfo sci{};
    sci.min_filter = SDL_GPU_FILTER_NEAREST;
    sci.mag_filter = SDL_GPU_FILTER_NEAREST;
    sci.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    sci.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sci.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sci.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    app->sampler = SDL_CreateGPUSampler(app->device, &sci);

    if (!app->noise || !app->h0 || !app->sampler)
    {
        SDL_Log("texture/sampler creation: %s", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    // fill then upload the gaussian noise
    // standard normals for both real and imaginary parts. marsaglia polar
    // same distribution barth's gpu box-muller produces apparently.
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

    // map -> memcpy -> unmap. SDL will keep the staging buffer alive until the copy pass finishes on the gpu
    void* mapped = SDL_MapGPUTransferBuffer(app->device, tb, false);
    SDL_memcpy(mapped, noisePixels.data(), tbci.size);
    SDL_UnmapGPUTransferBuffer(app->device, tb);

    SDL_GPUCommandBuffer* uploadCmd = SDL_AcquireGPUCommandBuffer(app->device);
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(uploadCmd);
    SDL_GPUTextureTransferInfo src{};
    src.transfer_buffer = tb;
    src.pixels_per_row = kN;
    src.rows_per_layer = kN;
    SDL_GPUTextureRegion region{};
    region.texture = app->noise;
    region.w = kN; region.h = kN; region.d = 1;
    SDL_UploadToGPUTexture(copyPass, &src, &region, false);
    SDL_EndGPUCopyPass(copyPass);
    SDL_SubmitGPUCommandBuffer(uploadCmd);
    SDL_ReleaseGPUTransferBuffer(app->device, tb);

    // dispatch the compute exactly once
    // h0(k) is static in step 2 - the time evolution that re-runs the
    // spectrum every frame is a future me problem.
    SDL_GPUCommandBuffer* computeCmd = SDL_AcquireGPUCommandBuffer(app->device);

    SDL_GPUStorageTextureReadWriteBinding rw{};
    rw.texture = app->h0;
    SDL_GPUComputePass* cpass = SDL_BeginGPUComputePass(computeCmd, &rw, 1, nullptr, 0);

    SDL_BindGPUComputePipeline(cpass, app->compute);

    SDL_GPUTextureSamplerBinding noiseBinding{};
    noiseBinding.texture = app->noise;
    noiseBinding.sampler = app->sampler;
    SDL_BindGPUComputeSamplers(cpass, 0, &noiseBinding, 1);

    // slight off-axis tilt so the wind-alignment shape isn't pixel-locked
    // to a row/column. normalising in the holy mfg library is cheap and happens
    // once, the shader assumes it's already unit length.
    const mfg::vec2 windDir = mfg::Normalize(mfg::vec2(1.0f, 0.28f));
    Params p{};
    p.N = kN;
    p.patchLength = kPatch;
    p.amplitude = kAmp;
    p.windSpeed = kWindSpeed;
    p.windX = windDir.x();
    p.windY = windDir.y();
    p.windPower = kWindPower;
    p.gravity = kGravity;
    SDL_PushGPUComputeUniformData(computeCmd, 0, &p, sizeof(p));

    // ceil(N/8) groups in each dim; the shader guards against overshoot.
    SDL_DispatchGPUCompute(cpass, (kN + 7) / 8, (kN + 7) / 8, 1);
    SDL_EndGPUComputePass(cpass);
    SDL_SubmitGPUCommandBuffer(computeCmd);

    return SDL_APP_CONTINUE;
}

// ==========================================================================
// SDL_AppIterate - frame by frame rendering
// ==========================================================================
SDL_AppResult SDL_AppIterate(void* appstate)
{
    App* app = static_cast<App*>(appstate);

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(app->device);

    // swapchain texture might come back null if the window is minimised or
    // mid-resize. submit the (empty) command buffer anyway to keep sdl happy and try again next tick.
    SDL_GPUTexture* swap = nullptr;
    Uint32 w = 0, h = 0;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd, app->window, &swap, &w, &h) || !swap)
    {
        SDL_SubmitGPUCommandBuffer(cmd);
        return SDL_APP_CONTINUE;
    }

    SDL_GPUColorTargetInfo ct{};
    ct.texture     = swap;
    ct.clear_color = { 12 / 255.0f, 12 / 255.0f, 18 / 255.0f, 1.0f };
    ct.load_op     = SDL_GPU_LOADOP_CLEAR;
    ct.store_op    = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &ct, 1, nullptr);
    SDL_BindGPUGraphicsPipeline(pass, app->draw);

    SDL_GPUTextureSamplerBinding h0Binding{};
    h0Binding.texture = app->h0;
    h0Binding.sampler = app->sampler;
    SDL_BindGPUFragmentSamplers(pass, 0, &h0Binding, 1);

    // fullscreen.vert assembles the triangle from gl_VertexIndex alone 
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
// BURN IT ALL TO THE GROUND SDL_AppQuit
// ==========================================================================
void SDL_AppQuit(void* appstate, SDL_AppResult /*result*/)
{
    App* app = static_cast<App*>(appstate);
    if (!app) { return; }

    if (app->device)
    {
        if (app->draw)    { SDL_ReleaseGPUGraphicsPipeline(app->device, app->draw); }
        if (app->compute) { SDL_ReleaseGPUComputePipeline(app->device, app->compute); }
        if (app->sampler) { SDL_ReleaseGPUSampler(app->device, app->sampler); }
        if (app->h0)      { SDL_ReleaseGPUTexture(app->device, app->h0); }
        if (app->noise)   { SDL_ReleaseGPUTexture(app->device, app->noise); }
        if (app->window)  { SDL_ReleaseWindowFromGPUDevice(app->device, app->window); }
        SDL_DestroyGPUDevice(app->device);
    }
    if (app->window) { SDL_DestroyWindow(app->window); }
    delete app;
}
