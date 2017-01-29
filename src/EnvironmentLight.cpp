#include <torch/EnvironmentLight.h>
#include <torch/Context.h>
#include <torch/Distribution.h>
#include <torch/Spectrum.h>
#include <torch/device/Light.h>

namespace torch
{

 const unsigned int EnvironmentLight::minRowCount = 3;

EnvironmentLight::EnvironmentLight(std::shared_ptr<Context> context) :
  Light(context),
  m_rowCount(minRowCount)
{
  Initialize();
}

unsigned int EnvironmentLight::GetRowCount() const
{
  return m_rowCount;
}

void EnvironmentLight::SetRowCount(unsigned int count)
{
  m_rowCount = std::max(minRowCount, count);
  UpdateDirectionCount();
  m_context->MarkDirty();
}

unsigned int EnvironmentLight::GetDirectionCount() const
{
  return m_directionCount;
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

void EnvironmentLight::SetRadiance(size_t index, float r, float g, float b)
{
  SetRadiance(index, Spectrum::FromRGB(r, g, b));
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
  // Light::BuildScene(link);

  // EnvironmentLightData data;
  // data.rowCount = m_rowCount;
  // data.radiance = reinterpret_cast<float3*>(m_radiance.data());

  // const optix::Matrix4x4 R = m_transform.GetRotationMatrix();
}

void EnvironmentLight::Initialize()
{
  UpdateDirectionCount();
}

void EnvironmentLight::UpdateDirectionCount()
{
  m_directionCount = 2;
  const float radiansPerRow = M_PIf / (m_rowCount - 1);

  for (unsigned int i = 1; i < m_rowCount - 1; ++i)
  {
    const float quadrantLength = 2 * M_PIf * sinf(i * radiansPerRow) / 4;
    m_directionCount += 4 * roundf(quadrantLength / radiansPerRow);
  }

  m_radiance.resize(m_directionCount);
}

} // namespace torch