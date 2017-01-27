#include <torch/AreaLight.h>
#include <torch/Context.h>
#include <torch/Geometry.h>

namespace torch
{

AreaLight::AreaLight(std::shared_ptr<Context> context) :
  Light(context)
{
}

Spectrum AreaLight::GetPower() const
{
  // TODO: use area
  return M_PIf * m_radiance; // * m_geometry->GetArea();
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
  // TODO: add to area light sampler
}

} // namespace torch