#version 460

/*
    stockham radix-4 fft, same one stage per dispatch as fft_stockham.comp,
    but running on RGBA32F textures with TWO complex numbers per pixel.

    why? the slope spectrum from time_spectrum.comp.glsl packs both the
    x slope and the z slope into one texture:
        .rg = i * k.x * h(k, t)     ->  d h / d x
        .ba = i * k.y * h(k, t)     ->  d h / d z

    the fft is linear, so i can transform both complex numbers
    independently in the same dispatch - same twiddle factors, same
    source coordinates, just two butterflies instead of one. cheaper
    than iffting each component with a separate rg32f pass.

    everything else (how stages work, src/dst addressing, the sign flag,
    the radix-4 butterfly math) is identical to fft_stockham.comp.glsl -
    see that file for the full explanation.
*/

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_src;
layout(set = 1, binding = 0, rgba32f) uniform writeonly image2D u_dst;

layout(set = 2, binding = 0) uniform FFTParams
{
    int   u_N;            // 0    grid size, power of four
    int   u_stage;        // 4    1..log4(N)
    int   u_horizontal;   // 8    1 = butterfly along x, 0 = along y
    float u_sign;         // 12   +1 inverse, -1 forward
};

const float TWO_PI = 6.28318530717958647692;

vec2 cmul(vec2 a, vec2 b)
{
    return vec2(a.x * b.x - a.y * b.y,
                a.x * b.y + a.y * b.x);
}

void main()
{
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    if (id.x >= u_N || id.y >= u_N) { return; }

    int i, fixedIdx;
    if (u_horizontal == 1) { i = id.x; fixedIdx = id.y; }
    else                   { i = id.y; fixedIdx = id.x; }

    int Lq       = 1 << (2 * (u_stage - 1));
    int L        = Lq << 2;
    int quarterN = u_N >> 2;

    int j    = i & (Lq - 1);
    int t    = i >> (2 * u_stage);
    int slot = (i >> (2 * (u_stage - 1))) & 3;

    int in0 = t * Lq + j;
    int in1 = in0 + quarterN;
    int in2 = in0 + quarterN * 2;
    int in3 = in0 + quarterN * 3;

    ivec2 c0 = (u_horizontal == 1) ? ivec2(in0, fixedIdx) : ivec2(fixedIdx, in0);
    ivec2 c1 = (u_horizontal == 1) ? ivec2(in1, fixedIdx) : ivec2(fixedIdx, in1);
    ivec2 c2 = (u_horizontal == 1) ? ivec2(in2, fixedIdx) : ivec2(fixedIdx, in2);
    ivec2 c3 = (u_horizontal == 1) ? ivec2(in3, fixedIdx) : ivec2(fixedIdx, in3);

    vec4 x0 = texelFetch(u_src, c0, 0);
    vec4 x1 = texelFetch(u_src, c1, 0);
    vec4 x2 = texelFetch(u_src, c2, 0);
    vec4 x3 = texelFetch(u_src, c3, 0);

    float angle = u_sign * TWO_PI * float(j) / float(L);
    vec2  w1 = vec2(cos(angle), sin(angle));
    vec2  w2 = cmul(w1, w1);
    vec2  w3 = cmul(w2, w1);

    // two independent radix-4 butterflies sharing the same twiddles:
    // one on .rg (x-slope), one on .ba (z-slope).
    vec2 z0_rg = x0.rg;
    vec2 z1_rg = cmul(w1, x1.rg);
    vec2 z2_rg = cmul(w2, x2.rg);
    vec2 z3_rg = cmul(w3, x3.rg);

    vec2 z0_ba = x0.ba;
    vec2 z1_ba = cmul(w1, x1.ba);
    vec2 z2_ba = cmul(w2, x2.ba);
    vec2 z3_ba = cmul(w3, x3.ba);

    vec2 A_rg = z0_rg + z2_rg;
    vec2 B_rg = z0_rg - z2_rg;
    vec2 C_rg = z1_rg + z3_rg;
    vec2 D_rg = z1_rg - z3_rg;

    vec2 A_ba = z0_ba + z2_ba;
    vec2 B_ba = z0_ba - z2_ba;
    vec2 C_ba = z1_ba + z3_ba;
    vec2 D_ba = z1_ba - z3_ba;

    // omega*D = sign * J * D = (-sign*D.y, sign*D.x)
    vec2 omegaD_rg = vec2(-u_sign * D_rg.y, u_sign * D_rg.x);
    vec2 omegaD_ba = vec2(-u_sign * D_ba.y, u_sign * D_ba.x);

    vec2 res_rg, res_ba;
    if (slot == 0)
    {
        res_rg = A_rg + C_rg;
        res_ba = A_ba + C_ba;
    }
    else if (slot == 1)
    {
        res_rg = B_rg + omegaD_rg;
        res_ba = B_ba + omegaD_ba;
    }
    else if (slot == 2)
    {
        res_rg = A_rg - C_rg;
        res_ba = A_ba - C_ba;
    }
    else
    {
        res_rg = B_rg - omegaD_rg;
        res_ba = B_ba - omegaD_ba;
    }

    imageStore(u_dst, id, vec4(res_rg, res_ba));
}
