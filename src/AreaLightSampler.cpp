#include <torch/AreaLightSampler.h>
#include <torch/Context.h>
#include <torch/Distribution.h>
#include <torch/PtxUtil.h>
#include <torch/device/Light.h>

namespace torch
{

AreaLightSampler::AreaLightSampler(std::shared_ptr<Context> context) :
  LightSampler(context)
{
  Initialize();
}

optix::Program AreaLightSampler::GetProgram() const
{
  return m_program;
}

float AreaLightSampler::GetLuminance() const
{
  float luminance = 0;

  for (const AreaLightData& light : m_lights)
  {
    luminance += light.luminance;
  }

  return luminance;
}

void AreaLightSampler::Clear()
{
  m_lights.clear();
}

void AreaLightSampler::Update()
{
  UpdateDistribution();
  UpdateLightBuffer();
}

void AreaLightSampler::Add(const AreaLightData& light)
{
  m_lights.push_back(light);
}

void AreaLightSampler::UpdateDistribution()
{
  std::vector<float> luminance;
  luminance.reserve(m_lights.size());

  for (const AreaLightData& light : m_lights)
  {
    luminance.push_back(light.luminance);
  }

  m_distribution->SetValues(luminance);
}

void AreaLightSampler::UpdateLightBuffer()
{
  AreaLightData* device;
  m_buffer->setSize(m_lights.size());
  device = reinterpret_cast<AreaLightData*>(m_buffer->map());
  std::copy(m_lights.begin(), m_lights.end(), device);
  m_buffer->unmap();
}

void AreaLightSampler::Initialize()
{
  CreateProgram();
  CreateDistribution();
  CreateLightBuffer();
}

void AreaLightSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("AreaLightSampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void AreaLightSampler::CreateDistribution()
{
  m_distribution = std::make_unique<Distribution>(m_context);
  m_program["GetLightIndex"]->set(m_distribution->GetProgram());
}

void AreaLightSampler::CreateLightBuffer()
{
  m_buffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["lights"]->setBuffer(m_buffer);
  m_buffer->setFormat(RT_FORMAT_USER);
  m_buffer->setElementSize(sizeof(AreaLightData));
  m_buffer->setSize(1);
}

} // namespace torch