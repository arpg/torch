#include <torch/AreaLight.h>
#include <torch/Context.h>
#include <torch/Geometry.h>
#include <torch/SceneLightSampler.h>
#include <torch/device/Light.h>

namespace torch
{

AreaLight::AreaLight(std::shared_ptr<Context> context) :
  Light(context)
{
}

Spectrum AreaLight::GetRadiance() const
{
  return m_radiance;
}

void AreaLight::SetRadiance(const Spectrum& radiance)
{
  m_radiance = radiance;
  m_context->MarkDirty();
}

void AreaLight::SetRadiance(float r, float g, float b)
{
  SetRadiance(Spectrum::FromRGB(r, g, b));
}

std::shared_ptr<Geometry> AreaLight::GetGeometry() const
{
  return m_geometry;
}

void AreaLight::SetGeometry(std::shared_ptr<Geometry> geometry)
{
  m_geometry = geometry;
  m_context->MarkDirty();
}

void AreaLight::BuildScene(Link& link)
{
  Light::BuildScene(link);

  AreaLightData data;
  data.geometry = 0;
  data.radiance = m_radiance.GetRGB();
  // data.luminance = GetLuminance();
  // data.area = m_geometry->GetArea();
  data.area = 1;

  std::shared_ptr<SceneLightSampler> sampler;
  sampler = m_context->GetLightSampler();
  sampler->Add(data);
}

} // namespace torch