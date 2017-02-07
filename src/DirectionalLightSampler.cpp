#include <torch/DirectionalLightSampler.h>
#include <torch/Context.h>
#include <torch/Distribution1D.h>
#include <torch/PtxUtil.h>
#include <torch/device/Light.h>

namespace torch
{

DirectionalLightSampler::DirectionalLightSampler(std::shared_ptr<Context> context) :
  LightSampler(context)
{
  Initialize();
}

optix::Program DirectionalLightSampler::GetProgram() const
{
  return m_program;
}

float DirectionalLightSampler::GetLuminance() const
{
  float luminance = 0;

  for (const DirectionalLightData& light : m_lights)
  {
    luminance += light.luminance;
  }

  return luminance;
}

void DirectionalLightSampler::Clear()
{
  m_lights.clear();
}

void DirectionalLightSampler::Update()
{
  UpdateDistribution();
  UpdateLightBuffer();
}

void DirectionalLightSampler::Add(const DirectionalLightData& light)
{
  m_lights.push_back(light);
}

void DirectionalLightSampler::UpdateDistribution()
{
  std::vector<float> luminance;
  luminance.reserve(m_lights.size());

  for (const DirectionalLightData& light : m_lights)
  {
    luminance.push_back(light.luminance);
  }

  m_distribution->SetValues(luminance);
}

void DirectionalLightSampler::UpdateLightBuffer()
{
  DirectionalLightData* device;
  m_buffer->setSize(m_lights.size());
  device = reinterpret_cast<DirectionalLightData*>(m_buffer->map());
  std::copy(m_lights.begin(), m_lights.end(), device);
  m_buffer->unmap();
}

void DirectionalLightSampler::Initialize()
{
  CreateProgram();
  CreateDistribution();
  CreateLightBuffer();
}

void DirectionalLightSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("DirectionalLightSampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void DirectionalLightSampler::CreateDistribution()
{
  m_distribution = std::make_unique<Distribution1D>(m_context);
  m_program["GetLightIndex"]->set(m_distribution->GetProgram());
}

void DirectionalLightSampler::CreateLightBuffer()
{
  m_buffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["lights"]->setBuffer(m_buffer);
  m_buffer->setFormat(RT_FORMAT_USER);
  m_buffer->setElementSize(sizeof(DirectionalLightData));
  m_buffer->setSize(1);
}

} // namespace torch