#include <torch/device/Camera.h>
#include <torch/device/Light.h>
#include <torch/device/Random.h>
#include <torch/device/Visibility.h>

typedef rtCallableProgramX<uint(float, float&)> Distribution1D;
typedef rtCallableProgramId<uint(float, float&)> Distribution1DId;

rtDeclareVariable(Distribution1D, GetLightIndex, , );
rtBuffer<Distribution1DId> SampleLight;

rtBuffer<torch::VoxelLightSubData> subdata;
rtBuffer<rtBufferId<float3, 1>, 1> radiance;

rtDeclareVariable(uint, computeVoxelDerivs, , );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtBuffer<torch::CameraData> cameras;
rtBuffer<torch::PixelSample> pixelSamples;
rtBuffer<rtBufferId<float3, 2>, 1> lightDerivatives;

TORCH_DEVICE uint3 GetIndices(uint light, uint voxel)
{
  uint3 indices;
  const uint3 dimensions = subdata[light].dimensions;
  indices.x = voxel % dimensions.x;
  indices.y = (voxel / dimensions.x) % dimensions.y;
  indices.z = (voxel / dimensions.x) / dimensions.y;
  return indices;
}

TORCH_DEVICE void GetVoxelCenter(uint light, uint voxel, float3& center)
{
  const float voxelSize = subdata[light].voxelSize;
  const uint3 dimensions = subdata[light].dimensions;
  const float3 gridSize = voxelSize * make_float3(dimensions);
  const uint3 indices = GetIndices(light, voxel);
  center = voxelSize * (make_float3(indices) + 0.5f) - gridSize / 2.0f;
  center = subdata[light].transform * make_float4(center, 1);
}

TORCH_DEVICE void SampleVoxel(const float3& origin, uint& seed,
    uint light, uint voxel, float3& point, float3& rad, float& pdf)
{
  GetVoxelCenter(light, voxel, point);
  float distance = length(point - origin);
  const float radius = subdata[light].voxelSize / 2.0f;
  const float area = 4 * M_PIf * radius * radius;
  rad = radiance[light][voxel] / (distance * area);

  const float3& x = origin;
  const float3& c = point;
  const float& r = subdata[light].voxelSize / 2;
  const float& E1 = torch::randf(seed);
  const float& E2 = torch::randf(seed);

  const float3 delta_xc = x - c;
  const float dist_xc = length(delta_xc);

  if (r > dist_xc)
  {
    rad = make_float3(0, 0, 0);
    pdf = 0.0f;
    return;
  }

  const float3 w = -normalize(delta_xc);
  float3 u1 = make_float3(0, -w.z, w.y);
  float3 u2 = make_float3(-w.z, 0, w.x);

  if (dot(u1, u1) > dot(u2, u2))
  {
    u1 = u1;
  }
  else
  {
    u1 = u2;
  }

  const float3 u = normalize(u1);
  const float3 v = cross(u, w);

  const float temp = r / dist_xc;
  const float cosAlpha = 1 - E1 + E1 * sqrtf(1 - (temp * temp));
  const float phi = 2 * M_PIf * E2;

  const float sinAlpha = sqrtf(1 - cosAlpha * cosAlpha);

  const float cosPhi_sinAlpha = cos(phi) * sinAlpha;
  const float sinPhi_sinAlpha = sin(phi) * sinAlpha;

  float3 direction;
  direction.x = u.x * cosPhi_sinAlpha + v.x * sinPhi_sinAlpha + w.x * cosAlpha;
  direction.y = u.y * cosPhi_sinAlpha + v.y * sinPhi_sinAlpha + w.y * cosAlpha;
  direction.z = u.z * cosPhi_sinAlpha + v.z * sinPhi_sinAlpha + w.z * cosAlpha;
  direction = normalize(direction);

  // compute intersect intermediate results
  const float od = dot(delta_xc, -direction);
  const float oo = dot(delta_xc, delta_xc);
  const float r2 = r * r;
  float a  = od * od - oo + r2;
  a = (a < 0) ? 0 : a;

  // compute intersects
  const float b = sqrt(a);
  const float t = -od - b;

  const float3 xp = x + t * -direction;
  const float3 norm = normalize(xp - c);
  const float cosTheta = dot(norm, direction);

  const float3 delta_xpx = xp - x;
  const float dist_xpx = length(delta_xpx);

  point = xp;

  if (isnan(point.x))
  {
    rtPrintf("Phi: %f\n", phi);
    rtPrintf("U:  %f %f %f\n", u.x, u.y, u.z);
    rtPrintf("V:  %f %f %f\n", v.x, v.y, v.z);
    rtPrintf("W:  %f %f %f\n", w.x, w.y, w.z);
    rtPrintf("A: %f\n", a);
    rtPrintf("OD: %f\n", -od);
    rtPrintf("B: %f\n", b);
    rtPrintf("X:  %f %f %f\n", x.x, x.y, x.z);
    rtPrintf("T:  %f\n", t);
    rtPrintf("D:  %f %f %f\n", -direction.x, -direction.y, -direction.z);
  }

  // TODO: compute pdf correctly
  // pdf = cosTheta / (2 * M_PIf * dist_xpx * dist_xpx * (1 - sqrtf(1 - temp * temp)));
  pdf = 1.0;
}

RT_CALLABLE_PROGRAM void Sample(torch::LightSample& sample)
{
  float rand = torch::randf(sample.seed);
  const uint light = GetLightIndex(rand, sample.pdf);

  float voxelPdf, pointPdf;
  rand = torch::randf(sample.seed);
  const uint voxel = SampleLight[light](rand, voxelPdf);

  float3 point;

  SampleVoxel(sample.origin, sample.seed, light, voxel, point,
      sample.radiance, pointPdf);

  if (pointPdf > 1E-6f)
  {

  sample.tmax = length(point - sample.origin);
  sample.direction = normalize(point - sample.origin);
  sample.pdf *= voxelPdf * pointPdf;

  if (isnan(sample.direction.x))
  {
    rtPrintf("NANS!!!\n");
  }

  const bool visible = torch::IsVisible(sample);
  if (!visible) sample.radiance = make_float3(0, 0, 0);

  if (computeVoxelDerivs && visible) // TODO: check if light has non-empty derivs
  {
    float distance = sample.tmax;
    const float radius = subdata[light].voxelSize / 2.0f;
    const float area = 4 * M_PIf * radius * radius;
    const float theta = dot(sample.direction, sample.snormal);
    const uint2 derivIndex = make_uint2(launchIndex.x, voxel);
    lightDerivatives[light][derivIndex] -= theta * sample.throughput / (sample.pdf * distance * area);
  }

  }
}