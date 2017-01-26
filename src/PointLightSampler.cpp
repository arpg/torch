#include <torch/PointLightSampler.h>
#include <torch/Context.h>
#include <torch/Distribution.h>
#include <torch/PtxUtil.h>

namespace torch
{

PointLightSampler::PointLightSampler(std::shared_ptr<Context> context) :
  LightSampler(context)
{
  Initialize();
}

PointLightSampler::~PointLightSampler()
{
}

optix::Program PointLightSampler::GetProgram() const
{
  return m_program;
}

float PointLightSampler::GetLuminance() const
{
  float luminance = 0;

  for (const PointLightData& light : m_lights)
  {
    luminance += light.luminance;
  }

  return luminance;
}

void PointLightSampler::Clear()
{
  m_lights.clear();
}

void PointLightSampler::Update()
{
  UpdateDistribution();
  UpdateLightBuffer();
}

void PointLightSampler::Add(const PointLightData& light)
{
  m_lights.push_back(light);
}

void PointLightSampler::UpdateDistribution()
{
  std::vector<float> luminance;
  luminance.reserve(m_lights.size());

  for (const PointLightData& light : m_lights)
  {
    luminance.push_back(light.luminance);
  }

  m_distribution->SetValues(luminance);
}

void PointLightSampler::UpdateLightBuffer()
{
  PointLightData* device;
  m_buffer->setSize(m_lights.size());
  device = reinterpret_cast<PointLightData*>(m_buffer->map());
  std::copy(m_lights.begin(), m_lights.end(), device);
  m_buffer->unmap();
}

void PointLightSampler::Initialize()
{
  CreateDistribution();
  CreateLightBuffer();
  CreateProgram();
}

void PointLightSampler::CreateDistribution()
{
  m_distribution = std::make_unique<Distribution>(m_context);
}

void PointLightSampler::CreateLightBuffer()
{
  m_buffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_buffer->setFormat(RT_FORMAT_USER);
  m_buffer->setElementSize(sizeof(PointLightData));
  m_buffer->setSize(1);
}

void PointLightSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("PointLightSampler");
  m_program = m_context->CreateProgram(file, "Sample");
  m_program["lights"]->setBuffer(m_buffer);
}

} // namespace torch