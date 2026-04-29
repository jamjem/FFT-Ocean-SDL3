#version 460

/*
    draws the spatial gradient (d h / d x, d h / d z) of the ocean.

    after the slope ifft, u_slope holds (per pixel):
        .r = Re(d h / d x)     .g = Im(d h / d x)  (~0)
        .b = Re(d h / d z)     .a = Im(d h / d z)  (~0)

    imaginary parts should be ~zero because the spectrum i fed in was
    built to be hermitian-symmetric (h0(-k) = h0(k)*), which guarantees
    a real output when you ifft it. if the imaginaries aren't tiny,
    something upstream is broken.

    my custom stockham ifft skips the 1/N^2 normalisation (forward fft
    multiplied by N^2, inverse should divide by N^2 to get back - i
    just never did), so the values coming in are N^2 times too big. i
    undo that here with invNN before mapping to colour.

    finally i show abs(slope) so both positive and negative slope
    magnitudes light up the same colour. that gives the dark-with-
    red/green-wave look from barth's reference: dark wherever the
    surface is flat, red where it's steep along x, green where it's
    steep along z, yellow where both are steep at once.
*/

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 FragColor;

// same binding slot as heightmap_view - swap shaders and bindings in
// the cpu code to display height instead of gradient.
layout(set = 2, binding = 0) uniform sampler2D u_slope;

void main()
{
    vec4 s = texture(u_slope, v_uv);

    // N=256 must match kN on the cpu side. if you resize the grid,
    // change both. no uniform buffer for one constant.
    const float invNN = 1.0 / (256.0 * 256.0);
    float dhdx = s.r * invNN;
    float dhdz = s.b * invNN;

    // any slope steeper than `range` clips to full red/green. with
    // L=1000, V=31 real slopes sit around 0.05-0.2, so a small ceiling
    // saturates the interesting features instead of burying them in
    // darkness. lower -> more contrast; raise if it all goes yellow.
    const float range = 0.15;
    float r = clamp(abs(dhdx) / range, 0.0, 1.0);
    float g = clamp(abs(dhdz) / range, 0.0, 1.0);

    FragColor = vec4(r, g, 0.0, 1.0);
}
