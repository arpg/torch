#include <torch/EnvironmentLight.h>
#include <torch/Context.h>
#include <torch/Distribution2D.h>
#include <torch/Link.h>
#include <torch/SceneLightSampler.h>
#include <torch/Spectrum.h>
#include <torch/device/Light.h>

namespace torch
{

EnvironmentLight::EnvironmentLight(std::shared_ptr<Context> context) :
  Light(context),
  m_rowCount(1)
{
  Initialize();
}

EnvironmentLight::~EnvironmentLight()
{
}

unsigned int EnvironmentLight::GetRowCount() const
{
  return m_rowCount;
}

void EnvironmentLight::SetRowCount(unsigned int count)
{
  m_rowCount = max(1, count);
  UpdateOffsets();
  m_context->MarkDirty();
}

unsigned int EnvironmentLight::GetDirectionCount() const
{
  return m_offsets.back();
}

unsigned int EnvironmentLight::GetDirectionCount(unsigned int row) const
{
  return m_offsets[row + 1] - m_offsets[row];
}

void EnvironmentLight::SetRadiance(const Spectrum& radiance)
{
  std::fill(m_radiance.begin(), m_radiance.end(), radiance);
  m_context->MarkDirty();
}

void EnvironmentLight::SetRadiance(float r, float g, float b)
{
  SetRadiance(Spectrum::FromRGB(r, g, b));
}

void EnvironmentLight::SetRadiance(size_t index, const Spectrum& radiance)
{
  m_radiance[index] = radiance;
  m_context->MarkDirty();
}

void EnvironmentLight::SetRadiance(const std::vector<Spectrum>& radiance)
{
  const size_t size = std::min(radiance.size(), m_radiance.size());
  std::copy(radiance.begin(), radiance.end() + size, m_radiance.data());
  m_context->MarkDirty();
}

void EnvironmentLight::SetDerivativeBuffer(optix::Buffer buffer)
{
  m_derivBuffer = buffer;
  m_context->MarkDirty();
}

void EnvironmentLight::BuildScene(Link& link)
{
  Light::BuildScene(link);
  UpdateDistribution();
  UpdateRadianceBuffer();
  UpdateOffsetBuffer();
  UpdateSampler(link);
}

optix::Buffer EnvironmentLight::GetRadianceBuffer() const
{
  return m_radianceBuffer;
}

optix::Buffer EnvironmentLight::GetOffsetBuffer() const
{
  return m_offsetBuffer;
}

void EnvironmentLight::UpdateDistribution()
{
  std::vector<float> luminances(m_radiance.size());

  for (size_t i = 0; i < luminances.size(); ++i)
  {
    luminances[i] = m_radiance[i].GetY();
  }

  m_distribution->SetValues(luminances, m_offsets);
}

void EnvironmentLight::UpdateRadianceBuffer()
{
  m_radianceBuffer->setSize(m_radiance.size());
  Spectrum* device = reinterpret_cast<Spectrum*>(m_radianceBuffer->map());
  std::copy(m_radiance.begin(), m_radiance.end(), device);
  m_radianceBuffer->unmap();
}

void EnvironmentLight::UpdateOffsetBuffer()
{
  m_offsetBuffer->setSize(m_offsets.size());
  unsigned int* device = reinterpret_cast<unsigned int*>(m_offsetBuffer->map());
  std::copy(m_offsets.begin(), m_offsets.end(), device);
  m_offsetBuffer->unmap();
}

void EnvironmentLight::UpdateSampler(Link& link)
{
  EnvironmentLightData data;
  data.distributionId = m_distribution->GetProgram()->getId();
  data.offsetsId = m_offsetBuffer->getId();
  data.radianceId = m_radianceBuffer->getId();
  data.derivId = (m_derivBuffer.get()) ? m_derivBuffer->getId() : 0;
  data.luminance = 1; // TODO: implement

  const optix::Matrix4x4 R =
      (link.GetTransform() * m_transform).GetRotationMatrix().inverse();

  data.rotation[0] = R[0];
  data.rotation[1] = R[1];
  data.rotation[2] = R[2];

  data.rotation[3] = R[4];
  data.rotation[4] = R[5];
  data.rotation[5] = R[6];

  data.rotation[6] = R[8];
  data.rotation[7] = R[9];
  data.rotation[8] = R[10];

  std::shared_ptr<SceneLightSampler> sampler;
  sampler = m_context->GetLightSampler();
  sampler->Add(data);
}

void EnvironmentLight::Initialize()
{
  CreateDistribution();
  CreateRadianceBuffer();
  CreateOffsetBuffer();
  UpdateOffsets();
}

void EnvironmentLight::CreateDistribution()
{
  m_distribution = std::make_unique<Distribution2D>(m_context);
}

void EnvironmentLight::CreateRadianceBuffer()
{
  m_radianceBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_radianceBuffer->setFormat(RT_FORMAT_FLOAT3);
  m_radianceBuffer->setSize(0);
}

void EnvironmentLight::CreateOffsetBuffer()
{
  m_offsetBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_offsetBuffer->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_offsetBuffer->setSize(0);
}

void EnvironmentLight::UpdateOffsets()
{
  const float radiansPerRow = M_PIf / (m_rowCount - 1);
  m_offsets.resize(m_rowCount + 1);
  m_offsets[0] = 0;
  m_offsets[1] = 1;

  for (unsigned int i = 1; i < m_rowCount - 1; ++i)
  {
    const float rowRadius = sinf(i * radiansPerRow);
    const float quadLength = (2 * M_PIf * rowRadius) / 4;
    const unsigned int directions = 4 * roundf(quadLength / radiansPerRow);
    m_offsets[i + 1] = directions + m_offsets[i];
  }

  m_offsets[m_rowCount] = 1 + m_offsets[m_rowCount - 1];
  m_radiance.resize(m_offsets.back());
}

} // namespace torch