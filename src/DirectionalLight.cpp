#include <torch/DirectionalLight.h>
#include <torch/Context.h>
#include <torch/Link.h>
#include <torch/Scene.h>
#include <torch/SceneLightSampler.h>
#include <torch/device/Light.h>

namespace torch
{

DirectionalLight::DirectionalLight(std::shared_ptr<Context> context) :
  Light(context)
{
}

Spectrum DirectionalLight::GetRadiance() const
{
 return m_radiance;
}

void DirectionalLight::SetRadiance(const Spectrum& radiance)
{
  m_radiance = radiance;
  m_context->MarkDirty();
}

void DirectionalLight::SetRadiance(float r, float g, float b)
{
  SetRadiance(Spectrum::FromRGB(r, g, b));
}

Vector DirectionalLight::GetDirection() const
{
  const optix::Matrix4x4 R = m_transform.GetRotationMatrix();
  const float4 col = R.getCol(1);
  return Vector(col.x, col.y, col.z);
}

void DirectionalLight::SetDirection(const Vector& direction)
{
  const Vector y = direction.Normalize();
  Vector x = y.Cross(Vector(1, 0, 0));
  Vector z = y.Cross(Vector(0, 0, 1));
  if (x * x > z * z) x = z;
  z = y.Cross(x).Normalize();
  x = y.Cross(z).Normalize();

  optix::Matrix3x3 R;
  R.setCol(0, make_float3(x.x, x.y, x.z));
  R.setCol(1, make_float3(y.x, y.y, y.z));
  R.setCol(2, make_float3(z.x, z.y, z.z));

  m_transform.SetRotation(R);
  m_context->MarkDirty();
}

void DirectionalLight::SetDirection(float x, float y, float z)
{
  SetDirection(Vector(x, y, z));
}

void DirectionalLight::BuildScene(Link& link)
{
  Light::BuildScene(link);

  DirectionalLightData data;
  const Transform transform = link.GetTransform() * m_transform;
  const optix::Matrix4x4 R = transform.GetRotationMatrix();
  data.direction = make_float3(R.getCol(1));

  const Vector rgb = m_radiance.GetRGB();
  data.radiance = make_float3(rgb.x, rgb.y, rgb.z);

  data.luminance = 1;
  // data.luminance = GetLuminance();

  std::shared_ptr<SceneLightSampler> sampler;
  sampler = m_context->GetLightSampler();
  sampler->Add(data);
}

} // namespace torch