#version 460

/*
    tessendorf time evolution - the thing that actually makes the ocean
    move.

    from the initial spectrum h0(k) and its conjugate twin h0(-k)*, the
    spectrum at any time t falls out of one closed-form expression:

        h(k, t) = h0(k) * exp(+i * w(k) * t)
                + h0(-k)* * exp(-i * w(k) * t)

    w(k) = sqrt(g * |k|) is the deep-water dispersion relation - it
    tells you how fast a wave with wave vector k oscillates in time. the
    two terms above are complex exponentials rotating in opposite
    directions; their sum gives me the real, back-and-forth wave
    behaviour i want.

    the nice thing is i don't have to simulate the intermediate frames -
    i jump straight to whatever t the cpu passes in. re-dispatching this
    shader every frame is way cheaper than re-computing phillips or
    re-randomising the gaussian noise.

    i also produce the SLOPE spectrum in the same pass. calculus trick:
    taking d/dx of h(x, t) = sum_k h(k, t) exp(+i k.x) pulls an (+i * k)
    out of every term, so:

        slope_x(k, t) = i * k.x * h(k, t)
        slope_z(k, t) = i * k.y * h(k, t)

    (my "z" in world space is the texture's y axis). packing both
    complex numbers into a single RGBA32F texture means i can ifft them
    together with one vec4-flavoured pass, instead of running two
    separate rg32f iffts.

    dispatched every frame, one gpu thread per texel (8x8 workgroups).
*/

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// packed h0: rg = h0(k), ba = h0(-k)*. sampled (read-only).
layout(set = 0, binding = 0) uniform sampler2D u_h0;

// storage images (raw write): rg = h(k, t), rgba = slope spectrum.
layout(set = 1, binding = 0, rg32f)   uniform writeonly image2D u_h;
layout(set = 1, binding = 1, rgba32f) uniform writeonly image2D u_slope;

layout(set = 2, binding = 0) uniform TimeParams
{
    int   u_N;
    float u_patchLength;  // L in meters
    float u_time;         // seconds
    float u_gravity;      // 9.81
};

// map a bin index (0..N-1) to the signed wave vector k in physical
// units. upper half of the grid represents negative frequencies (this
// is how the dft lays them out). scale by 2*pi/L to get radians/metre.
vec2 kVector(ivec2 bin)
{
    ivec2 m = ivec2(bin.x <= u_N / 2 ? bin.x : bin.x - u_N,
                    bin.y <= u_N / 2 ? bin.y : bin.y - u_N);
    return vec2(m) * (6.28318530717958647692 / u_patchLength);
}

// complex multiply: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
vec2 cmul(vec2 a, vec2 b)
{
    return vec2(a.x * b.x - a.y * b.y,
                a.x * b.y + a.y * b.x);
}

void main()
{
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    if (id.x >= u_N || id.y >= u_N) { return; }

    vec4 packed  = texelFetch(u_h0, id, 0);
    vec2 h0k     = packed.rg;   // h0(k)
    vec2 h0negC  = packed.ba;   // h0(-k)*

    // deep-water dispersion. wavelength -> angular frequency, without
    // worrying about water depth because i assume infinite ocean.
    vec2  k     = kVector(id);
    float omega = sqrt(u_gravity * length(k));
    float wt    = omega * u_time;

    // exp(+i*wt) = cos(wt) + i*sin(wt)
    // exp(-i*wt) = cos(wt) - i*sin(wt)
    float c = cos(wt);
    float s = sin(wt);
    vec2 ePos = vec2(c,  s);
    vec2 eNeg = vec2(c, -s);

    vec2 h = cmul(h0k, ePos) + cmul(h0negC, eNeg);

    imageStore(u_h, id, vec4(h, 0.0, 0.0));

    // slope = i * k * h. multiplying a complex number by i is just a
    // 90-degree rotation: i*(a + bi) = -b + ai. so i don't need a full
    // complex multiply - swap components, negate one, then scale by k.
    vec2 ih     = vec2(-h.y, h.x);      // i * h
    vec2 slopeX = k.x * ih;             // d h / d x  in frequency
    vec2 slopeZ = k.y * ih;             // d h / d z  in frequency
    imageStore(u_slope, id, vec4(slopeX, slopeZ));
}
