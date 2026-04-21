@echo off
:: compile every glsl file in this folder to spirv.
:: assumes glslc is on PATH (comes with the Vulkan SDK).
pushd %~dp0
glslc -fshader-stage=compute  initial_spectrum.comp.glsl -o initial_spectrum.comp.spv || goto :fail
glslc -fshader-stage=vertex   fullscreen.vert.glsl       -o fullscreen.vert.spv       || goto :fail
glslc -fshader-stage=fragment spectrum_view.frag.glsl    -o spectrum_view.frag.spv    || goto :fail
echo all shaders compiled
popd & exit /b 0
:fail
echo shader compile failed
popd & exit /b 1
