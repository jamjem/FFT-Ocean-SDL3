#version 460

/*
    tessendorf's h0(k) - the "initial spectrum".

    phillips P(k) says how much amplitude each wavevector should carry,
    but it's deterministic. slap a complex gaussian on it so every k
    gets an independent random amp + phase:

        h0(k) = (1/sqrt(2)) * sqrt(P(k)) * (xi_r + i * xi_i)

    dispatched once during init. writes rg of the output image.
    ba are left at zero - step 3 will pack the conjugate h0(-k)* in
    there for the time-evolution pass, but we don't need it yet.
*/

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// cpu-side gaussian noise, one N(0,1) per channel
layout(set = 0, binding = 0) uniform sampler2D u_noise;

// output: h0 in rg
layout(set = 1, binding = 0, rg32f) uniform writeonly image2D u_h0;

// must match the Params struct on the cpu side byte-for-byte (std140)
layout(set = 2, binding = 0) uniform Params
{
    int   u_N;            // texture resolution (N x N)
    float u_patchLength;  // L in meters - the ocean tile size
    float u_amplitude;    // A in phillips
    float u_windSpeed;    // |w| in m/s
    vec2  u_windDir;      // already normalised by the cpu
    float u_windPower;    // exponent p on |k_hat . w_hat|
    float u_gravity;      // 9.81
};

// fft bins run 0..N-1 but wave vectors need to be centered on 0.
// bins past N/2 wrap into negative land. multiply by 2pi/L to get
// the physical k in radians per meter.
vec2 kVector(ivec2 bin)
{
    ivec2 m = ivec2(bin.x <= u_N / 2 ? bin.x : bin.x - u_N,
                    bin.y <= u_N / 2 ? bin.y : bin.y - u_N);
    return vec2(m) * (6.28318530717958647692 / u_patchLength);
}

// phillips spectrum, same formula from step 1:
//     P(k) = A * exp(-1/(k*L_wave)^2) / k^4 * |k_hat . w_hat|^p
//
// 1/k^4 pumps energy into long waves, exp(-1/(k*L_wave)^2) kills the
// too-long ones, |k_hat . w_hat|^p is the wind-alignment factor.
// L_wave = V^2 / g is the biggest wave the wind can whip up, NOT the
// patch size. subtle and easy to screw up.
float phillips(vec2 k)
{
    float kLen = length(k);
    if (kLen < 1e-6) { return 0.0; }

    float k2 = kLen * kLen;
    float k4 = k2 * k2;

    float L  = (u_windSpeed * u_windSpeed) / u_gravity;
    float L2 = L * L;

    // abs'd and raised to p. assumes u_windDir is a unit vector.
    float align = pow(abs(dot(k / kLen, u_windDir)), u_windPower);

    return u_amplitude * (exp(-1.0 / (k2 * L2)) / k4) * align;
}

void main()
{
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    if (id.x >= u_N || id.y >= u_N) { return; }

    vec2  k  = kVector(id);
    float P  = phillips(k);
    vec2  xi = texelFetch(u_noise, id, 0).rg;

    // P(k) is variance, sqrt(P) turns it back into amplitude.
    // the 1/sqrt(2) is tessendorf's normalisation.
    const float INV_SQRT2 = 0.70710678118654752440;
    vec2 h0 = INV_SQRT2 * sqrt(max(P, 0.0)) * xi;

    imageStore(u_h0, id, vec4(h0, 0.0, 0.0));
}
