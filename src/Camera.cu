#include <optix.h>

rtDeclareVariable(uint2, pixelIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, imageSize, rtLaunchDim, );
rtBuffer<float3, 2> buffer;

RT_PROGRAM void Capture()
{
  buffer[pixelIndex] = make_float3(1, 0, 0);
}