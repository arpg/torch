#include <torch/MatteMaterial.h>
#include <torch/Context.h>

namespace torch
{

MatteMaterial::MatteMaterial(std::shared_ptr<Context> context) :
  Material(context, "MatteMaterial")
{
  Initialize();
}

void MatteMaterial::SetAlbedo(const Spectrum& albedo)
{
  m_albedos.resize(1);
  m_albedos[0] = albedo;
  UploadAlbedos();
}

void MatteMaterial::SetAlbedo(float r, float g, float b)
{
  SetAlbedo(Spectrum::FromRGB(r, g, b));
}

void MatteMaterial::SetAlbedos(const std::vector<Spectrum>& albedos)
{
  m_albedos = albedos;
  UploadAlbedos();
}

void MatteMaterial::Initialize()
{
  CreateAlbedoBuffer();
  UploadAlbedos();
}

void MatteMaterial::CreateAlbedoBuffer()
{
  m_albedos.push_back(Spectrum());
  m_albedoBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_albedoBuffer->setFormat(RT_FORMAT_FLOAT3);
  m_albedoBuffer->setSize(1);
  m_material["albedos"]->setBuffer(m_albedoBuffer);
}

void MatteMaterial::UploadAlbedos()
{
  m_albedoBuffer->setSize(m_albedos.size());
  Spectrum* device = reinterpret_cast<Spectrum*>(m_albedoBuffer->map());
  std::copy(m_albedos.begin(), m_albedos.end(), device);
  m_albedoBuffer->unmap();
}

} // namespace torch