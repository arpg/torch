#include <optix.h>
#include <torch/Ray.h>

rtDeclareVariable(float3, albedo, , );
rtDeclareVariable(torch::RadianceData, rayData, rtPayload, );

RT_PROGRAM void ClosestHit()
{
  rayData.radiance += rayData.throughput * albedo;
}