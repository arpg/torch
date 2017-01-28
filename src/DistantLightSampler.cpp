#include <torch/DistantLightSampler.h>
#include <torch/Context.h>
#include <torch/Distribution.h>
#include <torch/LightData.h>
#include <torch/PtxUtil.h>

namespace torch
{

DistantLightSampler::DistantLightSampler(std::shared_ptr<Context> context) :
  LightSampler(context)
{
  Initialize();
}

optix::Program DistantLightSampler::GetProgram() const
{
  return m_program;
}

float DistantLightSampler::GetLuminance() const
{
  float luminance = 0;

  for (const DistantLightData& light : m_lights)
  {
    luminance += light.luminance;
  }

  return luminance;
}

void DistantLightSampler::Clear()
{
  m_lights.clear();
}

void DistantLightSampler::Update()
{
  UpdateDistribution();
  UpdateLightBuffer();
}

void DistantLightSampler::Add(const DistantLightData& light)
{
  m_lights.push_back(light);
}

void DistantLightSampler::UpdateDistribution()
{
  std::vector<float> luminance;
  luminance.reserve(m_lights.size());

  for (const DistantLightData& light : m_lights)
  {
    luminance.push_back(light.luminance);
  }

  m_distribution->SetValues(luminance);
}

void DistantLightSampler::UpdateLightBuffer()
{
  DistantLightData* device;
  m_buffer->setSize(m_lights.size());
  device = reinterpret_cast<DistantLightData*>(m_buffer->map());
  std::copy(m_lights.begin(), m_lights.end(), device);
  m_buffer->unmap();
}

void DistantLightSampler::Initialize()
{
  CreateProgram();
  CreateDistribution();
  CreateLightBuffer();
}

void DistantLightSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("DistantLightSampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void DistantLightSampler::CreateDistribution()
{
  m_distribution = std::make_unique<Distribution>(m_context);
  m_program["GetLightIndex"]->set(m_distribution->GetProgram());
}

void DistantLightSampler::CreateLightBuffer()
{
  m_buffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["lights"]->setBuffer(m_buffer);
  m_buffer->setFormat(RT_FORMAT_USER);
  m_buffer->setElementSize(sizeof(DistantLightData));
  m_buffer->setSize(1);
}

} // namespace torch