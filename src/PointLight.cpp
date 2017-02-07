#include <torch/PointLight.h>
#include <torch/Context.h>
#include <torch/Scene.h>
#include <torch/SceneLightSampler.h>
#include <torch/Link.h>
#include <torch/Transform.h>
#include <torch/device/Light.h>

namespace torch
{

PointLight::PointLight(std::shared_ptr<Context> context) :
  Light(context)
{
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

  PointLightData data;
  const Transform transform = link.GetTransform() * m_transform;

  const Vector t = transform.GetTranslation();
  data.position = make_float3(t.x, t.y, t.z);

  const Vector rgb = m_intensity.GetRGB();
  data.intensity = make_float3(rgb.x, rgb.y, rgb.z);

  data.luminance = 1; // GetLuminance();

  std::shared_ptr<SceneLightSampler> sampler;
  sampler = m_context->GetLightSampler();
  sampler->Add(data);
}

} // namespace torch