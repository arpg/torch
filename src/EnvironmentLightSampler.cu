#include <torch/device/Light.h>
#include <torch/device/Random.h>

typedef rtCallableProgramX<unsigned int(float, float&)> Distribution1D;
typedef rtCallableProgramId<void(const float2&, uint2&, float&)> Distribution2D;

rtDeclareVariable(Distribution1D, GetLightIndex, , );
rtBuffer<Distribution2D> SampleLight;
rtBuffer<optix::Matrix3x3> transforms;
rtBuffer<float3> radiance;

TORCH_DEVICE void GetDirection(unsigned int light, unsigned int row,
    unsigned int col, float3& direction)
{

}

TORCH_DEVICE void GetDirection(unsigned int light, unsigned int row,
    unsigned int col, float3& direction)
{

}

TORCH_DEVICE void GetDirection(unsigned int light, unsigned int row,
    unsigned int col, float3& direction)
{

}

RT_CALLABLE_PROGRAM void Sample(torch::LightSample& sample)
{
  const float rand = torch::randf(sample.seed);
  const unsigned int lightIndex = GetLightIndex(rand, sample.pdf);

  float dirPdf;
  uint2 dirIndex;
  const float2 uv = torch::randf2(sample.seed);
  SampleLight[lightIndex](uv, dirIndex, dirPdf);
  GetDirection(lightIndex, dirIndex.x, dirIndex.y, sample.direction);

  sample.direction = transforms[lightIndex] * sample.direction;
  sample.direction = normalize(sample.direction);
  sample.radiance = radiance[radIndex];
  sample.tmax = RT_DEFAULT_MAX;
  sample.pdf *= dirPdf;
}