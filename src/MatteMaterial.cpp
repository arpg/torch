#include <torch/MatteMaterial.h>
#include <torch/Context.h>
#include <torch/device/Camera.h>

namespace torch
{

MatteMaterial::MatteMaterial(std::shared_ptr<Context> context) :
  Material(context, "MatteMaterial")
{
  Initialize();
}

size_t MatteMaterial::GetAlbedoCount() const
{
  return m_albedos.size();
}

void MatteMaterial::GetAlbedos(std::vector<Spectrum>& albedos) const
{
  albedos = m_albedos;
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

void MatteMaterial::SetDerivativeProgram(optix::Program program)
{
  m_material["AddToAlbedoJacobian"]->setProgramId(program);
}

void MatteMaterial::SetCameraBuffer(optix::Buffer buffer)
{
  m_cameraBuffer = buffer;
  m_material["cameras"]->setBuffer(m_cameraBuffer);
}

void MatteMaterial::SetPixelBuffer(optix::Buffer buffer)
{
  m_pixelBuffer = buffer;
  m_material["pixelSamples"]->setBuffer(m_pixelBuffer);
}

optix::Buffer MatteMaterial::GetAlbedoBuffer() const
{
  return m_albedoBuffer;
}

void MatteMaterial::Initialize()
{
  CreateCameraBuffer();
  CreatePixelBuffer();
  CreateAlbedoBuffer();
  UploadAlbedos();
}

void MatteMaterial::CreateCameraBuffer()
{
  m_cameraBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_cameraBuffer->setFormat(RT_FORMAT_USER);
  m_cameraBuffer->setElementSize(sizeof(CameraData));
  m_cameraBuffer->setSize(0);
  m_material["cameras"]->setBuffer(m_cameraBuffer);
}

void MatteMaterial::CreatePixelBuffer()
{
  m_pixelBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_pixelBuffer->setFormat(RT_FORMAT_USER);
  m_pixelBuffer->setElementSize(sizeof(PixelSample));
  m_pixelBuffer->setSize(0);
  m_material["pixelSamples"]->setBuffer(m_pixelBuffer);
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