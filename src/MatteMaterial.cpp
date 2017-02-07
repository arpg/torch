#include <torch/MatteMaterial.h>

namespace torch
{

MatteMaterial::MatteMaterial(std::shared_ptr<Context> context) :
  Material(context, "MatteMaterial")
{
  Initialize();
}

Spectrum MatteMaterial::GetAlbedo() const
{
  return m_albedo;
}

void MatteMaterial::SetAlbedo(const Spectrum& albedo)
{
  m_albedo = albedo;
  const Vector rgb = m_albedo.GetRGB();
  m_material["albedo"]->setFloat(rgb.x, rgb.y, rgb.z);
}

void MatteMaterial::SetAlbedo(float r, float g, float b)
{
  SetAlbedo(Spectrum::FromRGB(r, g, b));
}

void MatteMaterial::Initialize()
{
  const Vector rgb = m_albedo.GetRGB();
  m_material["albedo"]->setFloat(rgb.x, rgb.y, rgb.z);
}

} // namespace torch