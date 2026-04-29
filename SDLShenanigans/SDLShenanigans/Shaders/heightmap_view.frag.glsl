#version 460

/*
    draws the spatial ocean heightmap.

    after two separable ifft passes over h(k, t) the texture holds real
    heights in .r (.g should be ~0 from the hermitian symmetry of
    h(k, t) - if it's not, something upstream is broken). i remap
    [-range, +range] metres to greyscale, centred on 0 so troughs go
    black and peaks go white.

    no fftshift this time: for the spectrum view i shifted the uv to
    centre k=0 in the window, but the spatial heightmap is periodic
    over the tile so pixel (0,0) just lands in the corner and the ocean
    tiles seamlessly. fine for now.
*/

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 FragColor;

layout(set = 2, binding = 0) uniform sampler2D u_height;

void main()
{
    float h = texture(u_height, v_uv).r;

    // expected peak height with the default knobs (A=1, L=1000, V=31)
    // sits somewhere around 5-15 m. range clips anything beyond; lower
    // it for more contrast, raise it if whites/blacks are clipping.
    const float range = 15.0;
    float v = clamp(h / range * 0.5 + 0.5, 0.0, 1.0);

    FragColor = vec4(v, v, v, 1.0);
}
