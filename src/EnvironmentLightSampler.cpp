#include <torch/EnvironmentLightSampler.h>
#include <torch/Context.h>
#include <torch/Distribution1D.h>
#include <torch/Distribution2D.h>
#include <torch/PtxUtil.h>
#include <torch/Spectrum.h>
#include <torch/device/Light.h>

namespace torch
{

EnvironmentLightSampler::EnvironmentLightSampler(std::shared_ptr<Context> context) :
  LightSampler(context)
{
  Initialize();
}

EnvironmentLightSampler::~EnvironmentLightSampler()
{
}

optix::Program EnvironmentLightSampler::GetProgram() const
{
  return m_program;
}

float EnvironmentLightSampler::GetLuminance() const
{
  float luminance = 0;

  for (const EnvironmentLightData& light : m_lights)
  {
    luminance += light.luminance;
  }

  return luminance;
}

void EnvironmentLightSampler::Add(const EnvironmentLightData& light)
{
  m_lights.push_back(light);
}

void EnvironmentLightSampler::Clear()
{
  m_lights.clear();
}

void EnvironmentLightSampler::Update()
{
  UpdateDistribution();
  UpdateBuffers();
}

void EnvironmentLightSampler::UpdateDistribution()
{
  std::vector<float> luminance;
  luminance.reserve(m_lights.size());

  for (const EnvironmentLightData& light : m_lights)
  {
    luminance.push_back(light.luminance);
  }

  m_distribution->SetValues(luminance);
}

void EnvironmentLightSampler::UpdateBuffers()
{
  std::vector<int> samplePrograms(m_lights.size());
  std::vector<optix::Matrix3x3> rotations(m_lights.size());
  std::vector<unsigned int> offsets(m_lights.size());
  std::vector<int> radiance(m_lights.size());

  for (size_t i = 0; i < m_lights.size(); ++i)
  {
    samplePrograms[i] = m_lights[i].distributionId;
    rotations[i] = m_lights[i].rotation;
    offsets[i] = m_lights[i].offsetsId;
    radiance[i] = m_lights[i].radianceId;
  }

  WriteBuffer(m_samplePrograms, samplePrograms);
  WriteBuffer(m_rotations, rotations);
  WriteBuffer(m_offsets, offsets);
  WriteBuffer(m_radiance, radiance);
}

template <typename T>
void EnvironmentLightSampler::WriteBuffer(optix::Buffer buffer,
    const std::vector<T>& data)
{
  buffer->setSize(data.size());
  T* device = reinterpret_cast<T*>(buffer->map());
  std::copy(data.begin(), data.end(), device);
  buffer->unmap();
}

void EnvironmentLightSampler::Initialize()
{
  CreateProgram();
  CreateDistribution();
  CreateSampleProgramsBuffer();
  CreateOffsetBuffer();
  CreateRotationsBuffer();
  CreateRadianceBuffer();
}

void EnvironmentLightSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("EnvironmentLightSampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void EnvironmentLightSampler::CreateDistribution()
{
  m_distribution = std::make_unique<Distribution1D>(m_context);
  m_program["GetLightIndex"]->set(m_distribution->GetProgram());
}

void EnvironmentLightSampler::CreateSampleProgramsBuffer()
{
  m_samplePrograms = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["SampleLight"]->setBuffer(m_samplePrograms);
  m_samplePrograms->setFormat(RT_FORMAT_PROGRAM_ID);
  m_samplePrograms->setSize(0);
}

void EnvironmentLightSampler::CreateOffsetBuffer()
{
  m_offsets = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["offsets"]->setBuffer(m_offsets);
  m_offsets->setFormat(RT_FORMAT_BUFFER_ID);
  m_offsets->setSize(0);
}

void EnvironmentLightSampler::CreateRotationsBuffer()
{
  m_rotations = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["rotations"]->setBuffer(m_rotations);
  m_rotations->setFormat(RT_FORMAT_USER);
  m_rotations->setElementSize(sizeof(optix::Matrix3x3));
  m_rotations->setSize(0);
}

void EnvironmentLightSampler::CreateRadianceBuffer()
{
  m_radiance = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["radiance"]->setBuffer(m_radiance);
  m_radiance->setFormat(RT_FORMAT_BUFFER_ID);
  m_radiance->setSize(0);
}

} // namespace torch