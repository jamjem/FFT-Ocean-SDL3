#version 460

/*
    oversized triangle that covers the whole viewport, no vertex buffer
    required. just draw 3 verts and let gl_VertexIndex pick which corner.
    uvs run 0..1 over the visible area, and reach 2 at the off-screen
    corners (which get clipped away).

        idx 0 -> pos (-1, -1)  uv (0, 0)
        idx 1 -> pos ( 3, -1)  uv (2, 0)
        idx 2 -> pos (-1,  3)  uv (0, 2)
*/

layout(location = 0) out vec2 v_uv;

void main()
{
    vec2 uv = vec2(float((gl_VertexIndex << 1) & 2),
                   float( gl_VertexIndex       & 2));
    v_uv = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
