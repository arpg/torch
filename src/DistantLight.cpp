#include <torch/DistantLight.h>
#include <torch/Context.h>
#include <torch/Link.h>
#include <torch/Scene.h>
#include <torch/SceneLightSampler.h>
#include <torch/device/Light.h>

namespace torch
{

DistantLight::DistantLight(std::shared_ptr<Context> context) :
  Light(context)
{
}

Spectrum DistantLight::GetPower() const
{
  const float worldRadius = m_context->GetSceneRadius();
  return M_PI * worldRadius * worldRadius * m_radiance;
}

Spectrum DistantLight::GetRadiance() const
{
 return m_radiance;
}

void DistantLight::SetRadiance(const Spectrum& radiance)
{
  m_radiance = radiance;
  m_context->MarkDirty();
}

void DistantLight::SetRadiance(float r, float g, float b)
{
  SetRadiance(Spectrum::FromRGB(r, g, b));
}

float3 DistantLight::GetDirection() const
{
  const optix::Matrix4x4 R = m_transform.GetRotationMatrix();
  return make_float3(R.getCol(1));
}

void DistantLight::SetDirection(const float3& direction)
{
  const float3 y = normalize(direction);
  float3 x = cross(y, make_float3(1, 0, 0));
  float3 z = cross(y, make_float3(0, 0, 1));
  if (dot(x, x) > dot(z, z)) x = z;
  z = normalize(cross(y, x));
  x = normalize(cross(y, z));

  optix::Matrix3x3 R;
  R.setCol(0, x);
  R.setCol(1, y);
  R.setCol(2, z);

  m_transform.SetRotation(R);
  m_context->MarkDirty();
}

void DistantLight::SetDirection(float x, float y, float z)
{
  SetDirection(make_float3(x, y, z));
}

void DistantLight::BuildScene(Link& link)
{
  Light::BuildScene(link);

  DistantLightData data;
  const Transform transform = link.GetTransform() * m_transform;
  const optix::Matrix4x4 R = transform.GetRotationMatrix();
  data.direction = make_float3(R.getCol(1));
  data.radiance = m_radiance.GetRGB();
  data.luminance = GetLuminance();

  std::shared_ptr<SceneLightSampler> sampler;
  sampler = m_context->GetLightSampler();
  sampler->Add(data);
}

} // namespace torch