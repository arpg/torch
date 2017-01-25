#include <torch/MatteMaterial.h>

namespace torch
{

MatteMaterial::MatteMaterial(std::shared_ptr<Context> context) :
  Material(context, "MatteMaterial")
{
  Initialize();
}

MatteMaterial::~MatteMaterial()
{
}

Spectrum MatteMaterial::GetAlbedo() const
{
  return m_albedo;
}

void MatteMaterial::SetAlbedo(const Spectrum& albedo)
{
  m_albedo = albedo;
  m_material["albedo"]->setFloat(m_albedo.GetRGB());
}

void MatteMaterial::SetAlbedo(float r, float g, float b)
{
  SetAlbedo(Spectrum::FromRGB(r, g, b));
}

void MatteMaterial::Initialize()
{
  m_material["albedo"]->setFloat(m_albedo.GetRGB());
}

} // namespace torch