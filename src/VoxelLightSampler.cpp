#include <torch/VoxelLightSampler.h>
#include <torch/Context.h>
#include <torch/Distribution1D.h>
#include <torch/PtxUtil.h>
#include <torch/device/Light.h>

namespace torch
{

VoxelLightSampler::VoxelLightSampler(std::shared_ptr<Context> context) :
  LightSampler(context)
{
  Initialize();
}

VoxelLightSampler::~VoxelLightSampler()
{
}

optix::Program VoxelLightSampler::GetProgram() const
{
  return m_program;
}

float VoxelLightSampler::GetLuminance() const
{
  float luminance = 0;

  for (const VoxelLightData& light : m_lights)
  {
    luminance += light.luminance;
  }

  return luminance;
}

void VoxelLightSampler::Add(const VoxelLightData& light)
{
  m_lights.push_back(light);
}

void VoxelLightSampler::Clear()
{
  m_lights.clear();
}

void VoxelLightSampler::Update()
{
  UpdateDistribution();
  UpdateBuffers();
}

void VoxelLightSampler::UpdateDistribution()
{
  std::vector<float> luminance;
  luminance.reserve(m_lights.size());

  for (const VoxelLightData& light : m_lights)
  {
    luminance.push_back(light.luminance);
  }

  m_distribution->SetValues(luminance);
}

void VoxelLightSampler::UpdateBuffers()
{
  std::vector<int> samplePrograms(m_lights.size());
  std::vector<VoxelLightSubData> subdata(m_lights.size());
  std::vector<int> radiance(m_lights.size());
  std::vector<int> derivatives(m_lights.size());

  for (size_t i = 0; i < m_lights.size(); ++i)
  {
    samplePrograms[i] = m_lights[i].distributionId;
    subdata[i].transform = m_lights[i].transform;
    subdata[i].dimensions = m_lights[i].dimensions;
    subdata[i].voxelSize = m_lights[i].voxelSize;
    radiance[i] = m_lights[i].radianceId;

    derivatives[i] = (m_lights[i].derivId) ?
      m_lights[i].derivId : m_emptyDerivs->getId();
  }

  WriteBuffer(m_samplePrograms, samplePrograms);
  WriteBuffer(m_subdata, subdata);
  WriteBuffer(m_radiance, radiance);
  WriteBuffer(m_derivatives, derivatives);
}

template <typename T>
void VoxelLightSampler::WriteBuffer(optix::Buffer buffer,
    const std::vector<T>& data)
{
  buffer->setSize(data.size());
  T* device = reinterpret_cast<T*>(buffer->map());
  std::copy(data.begin(), data.end(), device);
  buffer->unmap();
}

void VoxelLightSampler::Initialize()
{
  CreateProgram();
  CreateDistribution();
  CreateSubDataBuffer();
  CreateSampleProgramsBuffer();
  CreateRadianceBuffer();
  CreateDerivativeBuffer();
  CreateEmptyDerivativeBuffer();
}

void VoxelLightSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("VoxelLightSampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void VoxelLightSampler::CreateDistribution()
{
  m_distribution = std::make_unique<Distribution1D>(m_context);
  m_program["GetLightIndex"]->set(m_distribution->GetProgram());
}

void VoxelLightSampler::CreateSubDataBuffer()
{
  m_subdata = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["subdata"]->setBuffer(m_subdata);
  m_subdata->setFormat(RT_FORMAT_USER);
  m_subdata->setElementSize(sizeof(VoxelLightSubData));
  m_subdata->setSize(0);
}

void VoxelLightSampler::CreateSampleProgramsBuffer()
{
  m_samplePrograms = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["SampleLight"]->setBuffer(m_samplePrograms);
  m_samplePrograms->setFormat(RT_FORMAT_PROGRAM_ID);
  m_samplePrograms->setSize(0);
}

void VoxelLightSampler::CreateRadianceBuffer()
{
  m_radiance = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["radiance"]->setBuffer(m_radiance);
  m_radiance->setFormat(RT_FORMAT_BUFFER_ID);
  m_radiance->setSize(0);
}

void VoxelLightSampler::CreateDerivativeBuffer()
{
  m_derivatives = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["lightDerivatives"]->setBuffer(m_derivatives);
  m_derivatives->setFormat(RT_FORMAT_BUFFER_ID);
  m_derivatives->setSize(0);
}

void VoxelLightSampler::CreateEmptyDerivativeBuffer()
{
  m_emptyDerivs = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_emptyDerivs->setFormat(RT_FORMAT_FLOAT3);
  m_emptyDerivs->setSize(1, 1);
}

} // namespace torch