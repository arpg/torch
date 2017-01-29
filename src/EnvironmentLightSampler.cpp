#include <torch/EnvironmentLightSampler.h>
#include <torch/Context.h>
#include <torch/Distribution.h>
#include <torch/PtxUtil.h>
#include <torch/device/Light.h>

namespace torch
{

EnvironmentLightSampler::EnvironmentLightSampler(std::shared_ptr<Context> context) :
  LightSampler(context)
{
  Initialize();
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

void EnvironmentLightSampler::Clear()
{
  m_lights.clear();
}

void EnvironmentLightSampler::Update()
{
  UpdateDistribution();
  UpdateBuffers();
}

void EnvironmentLightSampler::Add(const EnvironmentLightData& light)
{
  m_lights.push_back(light);
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
  // std::vector<unsigned int> rowPrograms;
  // std::vector<unsigned int> colPrograms;
  // std::vector<unsigned int> rowOffsets;
  // std::vector<unsigned int> colOffsets;
  // std::vector<optix::Matrix3x3> rotations;
  // std::vector<float3> radiance;

  // for (const EnvironmentLightData& light : m_lights)
  // {
  //   rowOffsets.push_back(colPrograms.size());
  //   unsigned int radIndex = 0;

  //   for (unsigned int i = 0; i < light.rowCount; ++i)
  //   {
  //     colOffsets.push_back(radiance.size());
  //     colPrograms.push_back(light.colPrograms[i]);

  //     for (unsigned int j = 0; j < light.colCounts[i]; ++j)
  //     {
  //       radiance.push_back(light.radiance[radIndex++]);
  //     }
  //   }

  //   rowPrograms.push_back(light.rowProgram);
  //   rotations.push_back(light.rotation);
  // }

  // rowOffsets.push_back(rowPrograms.size());
  // colOffsets.push_back(colPrograms.size());
  // WriteBuffer(m_rowPrograms, rowPrograms);
  // WriteBuffer(m_colPrograms, colPrograms);
  // WriteBuffer(m_rowOffsets, rowOffsets);
  // WriteBuffer(m_colOffsets, colOffsets);
  // WriteBuffer(m_rotations, rotations);
  // WriteBuffer(m_radiance, radiance);
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
  CreateBuffers();
}

void EnvironmentLightSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("EnvironmentLightSampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void EnvironmentLightSampler::CreateDistribution()
{
  m_distribution = std::make_unique<Distribution>(m_context);
  m_program["GetLightIndex"]->set(m_distribution->GetProgram());
}

void EnvironmentLightSampler::CreateBuffers()
{
  CreateRowProgramsBuffer();
  CreateColProgramsBuffer();
  CreateRowOffsetsBuffer();
  CreateColOffsetsBuffer();
  CreateRotationsBuffer();
  CreateRadianceBuffer();
}

void EnvironmentLightSampler::CreateRowProgramsBuffer()
{
  m_rowPrograms = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["GetRow"]->setBuffer(m_rowPrograms);
  m_rowPrograms->setFormat(RT_FORMAT_PROGRAM_ID);
  m_rowPrograms->setSize(0);
}

void EnvironmentLightSampler::CreateColProgramsBuffer()
{
  m_colPrograms = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["GetCol"]->setBuffer(m_colPrograms);
  m_colPrograms->setFormat(RT_FORMAT_PROGRAM_ID);
  m_colPrograms->setSize(0);
}

void EnvironmentLightSampler::CreateRowOffsetsBuffer()
{
  m_rowOffsets = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["rowOffsets"]->setBuffer(m_rowOffsets);
  m_rowOffsets->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_rowOffsets->setSize(0);
}

void EnvironmentLightSampler::CreateColOffsetsBuffer()
{
  m_colOffsets = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["colOffsets"]->setBuffer(m_colOffsets);
  m_colOffsets->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_colOffsets->setSize(0);
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
  m_radiance->setFormat(RT_FORMAT_FLOAT3);
  m_radiance->setSize(0);
}

} // namespace torch