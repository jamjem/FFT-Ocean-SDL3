::quick and dirty UI for converting vertex/fragment shaders from glsl to spirv
::assumes glslc is in your environment variables/you have the Vulkan SDK installed

@echo off
set /a has_invalid = 0

:choose_file
cls
if %has_invalid%==1 (
    echo [31minvalid choice, please enter the number option you wish to use[0m
)


echo [7mPete's Quick and Dirty GLSL to SPV Shader Conversion UI![0m
echo .
echo [1] Vertex Shader
echo [2] Fragment Shader
echo [3] Exit
set /p choice=">"

if %choice%==1 (
    set %has_invalid%=0
    goto VertexConvert
)
if %choice%==2 (
    set %has_invalid%=0
    goto FragmentConvert
) 
if %choice%==3 (
    exit
) else (
    set %choice%=,
    set %has_invalid%=1
    goto choose_file
)


:VertexConvert
cls
set /p filename="Enter name of GLSL vertex shader file to convert to SPV: "
glslc -fshader-stage=vertex %filename%.glsl -o %filename%.spv
if not exist %filename%.spv (
    echo conversion failed, ensure the filename/path you provided is exact
    goto VertexConvert
) else (
    echo Conversion success!
    goto end
)


:FragmentConvert
cls
set /p filename="Enter name of GLSL fragment shader file to convert to SPV: "
glslc -fshader-stage=fragment %filename%.glsl -o %filename%.spv
if not exist %filename%.spv (
    echo conversion failed, ensure the filename/path you provided is exact
    goto FragmentConvert
) else (
    echo Conversion success!
    goto end
)

:end
pause