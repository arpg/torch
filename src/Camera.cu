#include <optix.h>

rtDeclareVariable(uint2, pixelIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, imageSize, rtLaunchDim, );

rtDeclareVariable(float3, position, , );
rtDeclareVariable(float3, center, , );
rtDeclareVariable(float3, u, , );
rtDeclareVariable(float3, v, , );
rtDeclareVariable(float3, w, , );
rtBuffer<float3, 2> buffer;

RT_PROGRAM void Capture()
{
  buffer[pixelIndex] = make_float3(1, 0, 0);
}