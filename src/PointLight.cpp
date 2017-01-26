#include <torch/PointLight.h>
#include <torch/Context.h>
#include <torch/Scene.h>
#include <torch/SceneLightSampler.h>

namespace torch
{

PointLight::PointLight(std::shared_ptr<Context> context) :
  Light(context)
{
}

PointLight::~PointLight()
{
}

Spectrum PointLight::GetPower() const
{
  return 4 * M_PIf * m_intensity;
}

Spectrum PointLight::GetIntensity() const
{
 return m_intensity;
}

void PointLight::SetIntensity(const Spectrum& intensity)
{
  m_intensity = intensity;
  m_context->MarkDirty();
}

void PointLight::SetIntensity(float r, float g, float b)
{
  SetIntensity(Spectrum::FromRGB(r, g, b));
}

void PointLight::BuildScene(Link& link)
{
  Light::BuildScene(link);
  std::shared_ptr<SceneLightSampler> sampler;
  sampler = m_context->GetLightSampler();
  // sampler->Add(data);
}

} // namespace torch