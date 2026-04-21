#version 460

/*
    draws the initial spectrum h0(k) to the window.

    fftshift in the sampler: the compute shader wrote k=(0,0) to pixel
    (0,0), but for a human it's nicer if k=(0,0) sits at the center of
    the image so the wind-aligned bowtie is obvious. we shift the UV by
    half a texture instead of rearranging the texture itself.

    h0 is complex (r = real, g = imag). we clip negatives to zero and
    map real->red, imag->green. because xi is zero-mean, ~1/4 of bins
    land in each quadrant of the sign plane, which gives the classic
    red/green/yellow/black speckle look from barth's screenshots.
*/

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 FragColor;

layout(set = 2, binding = 0) uniform sampler2D u_h0;

void main()
{
    vec2 uv = fract(v_uv + 0.5);
    vec2 h  = texture(u_h0, uv).rg;

    // h0 has huge dynamic range, most energy is near k=0. exposure +
    // gamma squeezes it into 0..1 so the off-axis waves are still
    // visible. tuned for the defaults N=256, L=1000, V=31.
    const float exposure = 500.0;
    const float gamma    = 0.55;

    vec2 vis = pow(clamp(max(h, vec2(0.0)) * exposure, 0.0, 1.0), vec2(gamma));
    FragColor = vec4(vis, 0.0, 1.0);
}
