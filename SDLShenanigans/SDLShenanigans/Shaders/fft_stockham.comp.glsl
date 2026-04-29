#version 460

/*
    stockham radix-4 fft, ONE butterfly stage per dispatch.

    why radix-4 over radix-2?
      each stage merges FOUR sub-DFTs instead of two, so we need
      log4(N) stages per axis instead of log2(N). for N=256 that is
      4 stages per axis (8 total dispatches per 2d ifft) - half the
      dispatches of the radix-2 version (which would be 16), same mathematical result.
      N must be a power of 4.

    butterfly math (size-4 DFT with prescaled samples):
      four source samples are N/4 apart in the source texture:
          x0 = src[in0],  x1 = src[in0 + N/4]
          x2 = src[in0 + N/2],  x3 = src[in0 + 3N/4]
      where in0 = t*Lq + j,  Lq = 4^(s-1),  j = i mod Lq.

      twiddle factors at phase j:
          w1 = exp(sign * 2pi*i * j / L),  L = 4*Lq
          w2 = w1 * w1,  w3 = w2 * w1

      prescale each odd-indexed source, then do two radix-2-like
      reductions (share the sub-expressions across all 4 outputs):
          z0 = x0,      z2 = w2*x2
          z1 = w1*x1,   z3 = w3*x3

          A = z0 + z2,   B = z0 - z2
          C = z1 + z3,   D = z1 - z3

      ω = exp(sign * 2pi*i / 4) = sign * J (J = imaginary unit).
      the 4 outputs depend on which slot (0..3) this invocation owns,
      pulled from bits [2s-1 : 2s-2] of i:
          slot 0: A + C
          slot 1: B + ω*D
          slot 2: A - C
          slot 3: B - ω*D

      note the sign of ω matters - for inverse (sign=+1) ω=+J, for
      forward (sign=-1) ω=-J. that's why we multiply by u_sign below
      rather than hard-coding one direction.

    sign:
      -1 -> forward fft,  +1 -> inverse. i use +1 here since the point
      of step 4 is going back to the spatial domain. NO 1/N^2
      normalisation is applied; tessendorf's h(x, t) is defined as
      sum_k h~(k, t) exp(+i k.x), i.e. the unnormalised inverse dft,
      so leaving the scale alone keeps the heightmap in physical meters.
*/

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_src;
layout(set = 1, binding = 0, rg32f) uniform writeonly image2D u_dst;

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

    // Lq = 4^(s-1) = size of each input sub-DFT
    // L  = 4*Lq    = size of each output sub-DFT
    int Lq       = 1 << (2 * (u_stage - 1));
    int L        = Lq << 2;
    int quarterN = u_N >> 2;

    int j    = i & (Lq - 1);                        // twiddle phase index
    int t    = i >> (2 * u_stage);                  // which output group
    int slot = (i >> (2 * (u_stage - 1))) & 3;      // which of the 4 outputs

    int in0 = t * Lq + j;
    int in1 = in0 + quarterN;
    int in2 = in0 + quarterN * 2;
    int in3 = in0 + quarterN * 3;

    ivec2 c0 = (u_horizontal == 1) ? ivec2(in0, fixedIdx) : ivec2(fixedIdx, in0);
    ivec2 c1 = (u_horizontal == 1) ? ivec2(in1, fixedIdx) : ivec2(fixedIdx, in1);
    ivec2 c2 = (u_horizontal == 1) ? ivec2(in2, fixedIdx) : ivec2(fixedIdx, in2);
    ivec2 c3 = (u_horizontal == 1) ? ivec2(in3, fixedIdx) : ivec2(fixedIdx, in3);

    vec2 x0 = texelFetch(u_src, c0, 0).rg;
    vec2 x1 = texelFetch(u_src, c1, 0).rg;
    vec2 x2 = texelFetch(u_src, c2, 0).rg;
    vec2 x3 = texelFetch(u_src, c3, 0).rg;

    // twiddle factors w1 = exp(sign * 2pi*i * j / L)
    float angle = u_sign * TWO_PI * float(j) / float(L);
    vec2  w1 = vec2(cos(angle), sin(angle));
    vec2  w2 = cmul(w1, w1);
    vec2  w3 = cmul(w2, w1);

    // prescale odd-indexed samples with twiddles, then radix-4 butterfly
    vec2 z0 = x0;
    vec2 z1 = cmul(w1, x1);
    vec2 z2 = cmul(w2, x2);
    vec2 z3 = cmul(w3, x3);

    vec2 A = z0 + z2;
    vec2 B = z0 - z2;
    vec2 C = z1 + z3;
    vec2 D = z1 - z3;

    // omegaD = ω * D  where  ω = sign * J  (J = +i; u_sign = +1 inverse, -1 forward).
    // J * (x, y) = (-y, x), so sign*J*(x,y) = (-sign*y, sign*x).
    vec2 omegaD = vec2(-u_sign * D.y, u_sign * D.x);

    vec2 result;
    if      (slot == 0) result = A + C;
    else if (slot == 1) result = B + omegaD;
    else if (slot == 2) result = A - C;
    else                result = B - omegaD;

    imageStore(u_dst, id, vec4(result, 0.0, 0.0));
}
