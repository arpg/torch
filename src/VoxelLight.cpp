#include <torch/VoxelLight.h>
#include <torch/Context.h>
#include <torch/Distribution1D.h>
#include <torch/Exception.h>
#include <torch/SceneLightSampler.h>
#include <torch/device/Light.h>

namespace torch
{

VoxelLight::VoxelLight(std::shared_ptr<Context> context) :
  Light(context),
  m_voxelSize(0.1),
  m_gridDimensions(make_uint3(1, 1, 1))
{
  Initialize();
}

VoxelLight::~VoxelLight()
{
}

unsigned int VoxelLight::GetVoxelCount() const
{
  return m_gridDimensions.x * m_gridDimensions.y * m_gridDimensions.z;
}

void VoxelLight::SetRadiance(const Spectrum& radiance)
{
  std::fill(m_radiance.begin(), m_radiance.end(), radiance);
  m_context->MarkDirty();
}

void VoxelLight::SetRadiance(float r, float g, float b)
{
  SetRadiance(Spectrum::FromRGB(r, g, b));
}

void VoxelLight::SetRadiance(size_t index, const Spectrum& radiance)
{
  m_radiance[index] = radiance;
  m_context->MarkDirty();
}

void VoxelLight::SetRadiance(const std::vector<Spectrum>& radiance)
{
  const size_t size = std::min(radiance.size(), m_radiance.size());
  std::copy(radiance.begin(), radiance.begin() + size, m_radiance.data());
  m_context->MarkDirty();
}

void VoxelLight::SetDimensions(uint x, uint y, uint z)
{
  m_gridDimensions = make_uint3(x, y, z);
  m_radiance.resize(GetVoxelCount());
  m_context->MarkDirty();
}

void VoxelLight::SetVoxelSize(float size)
{
  m_voxelSize = size;
  m_context->MarkDirty();
}

void VoxelLight::SetDerivativeBuffer(optix::Buffer buffer)
{
  m_derivBuffer = buffer;
  m_context->MarkDirty();
}

void VoxelLight::BuildScene(Link& link)
{
  Light::BuildScene(link);
  UpdateDistribution();
  UpdateRadianceBuffer();
  UpdateSampler(link);
}

optix::Buffer VoxelLight::GetRadianceBuffer() const
{
  return m_radianceBuffer;
}

void VoxelLight::UpdateDistribution()
{
  std::vector<float> luminances(m_radiance.size());

  for (size_t i = 0; i < luminances.size(); ++i)
  {
    luminances[i] = m_radiance[i].GetY();
  }

  m_distribution->SetValues(luminances);
}

void VoxelLight::UpdateRadianceBuffer()
{
  m_radianceBuffer->setSize(m_radiance.size());
  Spectrum* device = reinterpret_cast<Spectrum*>(m_radianceBuffer->map());
  std::copy(m_radiance.begin(), m_radiance.end(), device);
  m_radianceBuffer->unmap();
}

void VoxelLight::UpdateSampler(Link& link)
{
  VoxelLightData light;
  light.transform = m_transform.GetMatrix3x4();
  light.distributionId = m_distribution->GetProgram()->getId();
  light.radianceId = m_radianceBuffer->getId();
  light.derivId = (m_derivBuffer.get()) ? m_derivBuffer->getId() : 0;
  light.dimensions = m_gridDimensions;
  light.voxelSize = m_voxelSize;
  light.luminance = 1; // TODO: implement

  std::shared_ptr<SceneLightSampler> sampler;
  sampler = m_context->GetLightSampler();
  sampler->Add(light);
}

void VoxelLight::Initialize()
{
  CreateDistribution();
  CreateRadianceBuffer();
}

void VoxelLight::CreateDistribution()
{
  m_distribution = std::make_unique<Distribution1D>(m_context, true);
}

void VoxelLight::CreateRadianceBuffer()
{
  m_radianceBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_radianceBuffer->setFormat(RT_FORMAT_FLOAT3);
  m_radianceBuffer->setSize(0);
}

} // namespace torch