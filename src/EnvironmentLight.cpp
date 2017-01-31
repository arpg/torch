#include <torch/EnvironmentLight.h>
#include <torch/Context.h>
#include <torch/Distribution1D.h>
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
  const auto endIndex = radiance.begin() + size;
  std::copy(radiance.begin(), endIndex, m_radiance.begin());
  m_context->MarkDirty();
}

void EnvironmentLight::BuildScene(Link& link)
{
  Light::BuildScene(link);

  EnvironmentLightData data;
  data.rowCount = m_rowCount;

  data.luminance = 1; // TODO: implement

  // TODO: fix both potential memory issues
  // we are passing a pointer to member variable
  // EnvironmentLightSampler hold reference to this
  // if this object (EnvironmentLight) is destroyed, the pointer will be invalid
  //============================================================================
  data.radiance = reinterpret_cast<float3*>(m_radiance.data());
  data.offsets = m_offsets.data();
  //============================================================================

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
  UpdateOffsets();
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