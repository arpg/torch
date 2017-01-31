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
  UpdateLightDistributions();
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

void EnvironmentLightSampler::UpdateLightDistributions()
{
  m_lightDistributions.resize(m_lights.size());

  for (unsigned int i = 0; i < m_lights.size(); ++i)
  {
    const EnvironmentLightData& light = m_lights[i];
    const unsigned int dirCount = light.offsets[light.rowCount];
    std::vector<float> values(dirCount);

    std::vector<unsigned int> offsets(light.rowCount + 1);
    std::copy(light.offsets, light.offsets + offsets.size(), offsets.data());

    for (unsigned int j = 0; j < dirCount; ++j)
    {
      const float3& a = light.radiance[j];
      const Spectrum b = Spectrum::FromRGB(a.x, a.y, a.z);
      values[j] = b.GetY(); // TODO: scale by area
    }

    m_lightDistributions[i] = std::make_unique<Distribution2D>(m_context);
    m_lightDistributions[i]->SetValues(values, offsets);
  }
}

void EnvironmentLightSampler::UpdateBuffers()
{
  unsigned int radianceSize = 0;
  unsigned int rowOffsetsSize = 0;

  for (unsigned int i = 0; i < m_lights.size(); ++i)
  {
    const EnvironmentLightData& light = m_lights[i];
    radianceSize += light.offsets[light.rowCount];
    rowOffsetsSize += light.rowCount;
  }

  // m_samplePrograms->setSize(m_lights.size());

  // optix::callableProgramId<uint2(const float2&, float&)>* programs =
  //     static_cast<optix::callableProgramId<uint2(const float2&, float&)>*>(m_samplePrograms->map());

  std::vector<unsigned int> samplePrograms(m_lights.size());
  std::vector<unsigned int> lightOffsets(m_lights.size() + 1);
  std::vector<optix::Matrix3x3> rotations(m_lights.size());
  std::vector<unsigned int> rowOffsets(rowOffsetsSize);
  std::vector<float3> radiance(radianceSize);
  lightOffsets[0] = 0;

  unsigned int rowIndex = 0;
  unsigned int radIndex = 0;

  for (unsigned int i = 0; i < m_lights.size(); ++i)
  {
    const EnvironmentLightData& light = m_lights[i];
    lightOffsets[i + 1] = light.rowCount;
    rotations[i] = light.rotation;
    samplePrograms[i] = m_lightDistributions[i]->GetProgram()->getId();

    // programs[i] = optix::callableProgramId<uint2(const float2&, float&)>(
    //       m_lightDistributions[i]->GetProgram());

    // m_program["SampleLight2"]->setProgramId(m_lightDistributions[i]->GetProgram());

    const unsigned int dirCount = light.offsets[light.rowCount];
    float3* first = light.radiance;
    float3* last = light.radiance + dirCount; // TODO: copy correctly
    float3* result = &radiance[radIndex];
    std::copy(first, last, result);
    radIndex += dirCount;

    for (unsigned int j = 0; j < light.rowCount; ++j)
    {
      rowOffsets[rowIndex++] = lightOffsets[i] + light.offsets[j];
    }
  }

  // m_samplePrograms->unmap();

  WriteBuffer(m_samplePrograms, samplePrograms);
  WriteBuffer(m_lightOffsets, lightOffsets);
  WriteBuffer(m_rowOffsets, rowOffsets);
  WriteBuffer(m_rotations, rotations);
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
  CreateBuffers();
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

void EnvironmentLightSampler::CreateBuffers()
{
  CreateSampleProgramsBuffer();
  CreateLightOffsetsBuffer();
  CreateRowOffsetsBuffer();
  CreateRotationsBuffer();
  CreateRadianceBuffer();
}

void EnvironmentLightSampler::CreateSampleProgramsBuffer()
{
  m_samplePrograms = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["SampleLight"]->setBuffer(m_samplePrograms);
  m_samplePrograms->setFormat(RT_FORMAT_PROGRAM_ID);
  m_samplePrograms->setSize(0);
}

void EnvironmentLightSampler::CreateLightOffsetsBuffer()
{
  m_lightOffsets = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["lightOffsets"]->setBuffer(m_lightOffsets);
  m_lightOffsets->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_lightOffsets->setSize(0);
}

void EnvironmentLightSampler::CreateRowOffsetsBuffer()
{
  m_rowOffsets = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["rowOffsets"]->setBuffer(m_rowOffsets);
  m_rowOffsets->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_rowOffsets->setSize(0);
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