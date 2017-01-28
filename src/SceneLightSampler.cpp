#include <torch/SceneLightSampler.h>
#include <torch/AreaLightSampler.h>
#include <torch/Context.h>
#include <torch/DistantLightSampler.h>
#include <torch/Distribution.h>
#include <torch/PointLightSampler.h>
#include <torch/PtxUtil.h>
#include <torch/device/Light.h>

namespace torch
{

SceneLightSampler::SceneLightSampler(std::shared_ptr<Context> context) :
  m_context(context)
{
  Initialize();
}

optix::Program SceneLightSampler::GetProgram() const
{
  return m_program;
}

void SceneLightSampler::Add(const AreaLightData& light)
{
  LightSampler* sampler = m_samplers[LIGHT_TYPE_AREA].get();
  static_cast<AreaLightSampler*>(sampler)->Add(light);
}

void SceneLightSampler::Add(const DistantLightData& light)
{
  LightSampler* sampler = m_samplers[LIGHT_TYPE_DISTANT].get();
  static_cast<DistantLightSampler*>(sampler)->Add(light);
}

void SceneLightSampler::Add(const PointLightData& light)
{
  LightSampler* sampler = m_samplers[LIGHT_TYPE_POINT].get();
  static_cast<PointLightSampler*>(sampler)->Add(light);
}

void SceneLightSampler::Clear()
{
  for (std::unique_ptr<LightSampler>& sampler : m_samplers)
  {
    sampler->Clear();
  }
}

void SceneLightSampler::Update()
{
  UpdateLightSamplers();
  UpdateDistribution();
}

void SceneLightSampler::UpdateLightSamplers()
{
  for (std::unique_ptr<LightSampler>& sampler : m_samplers)
  {
    sampler->Update();
  }
}

void SceneLightSampler::UpdateDistribution()
{
  std::vector<float> values;
  values.reserve(m_samplers.size());

  for (std::unique_ptr<LightSampler>& sampler : m_samplers)
  {
    values.push_back(sampler->GetLuminance());
  }

  m_distribution->SetValues(values);
}

void SceneLightSampler::Initialize()
{
  CreateProgram();
  CreateDistribution();
  CreateLightSamplers();
}

void SceneLightSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("SceneLightSampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void SceneLightSampler::CreateDistribution()
{
  m_distribution = std::make_unique<Distribution>(m_context);
  m_program["GetLightType"]->set(m_distribution->GetProgram());
}

void SceneLightSampler::CreateLightSamplers()
{
  m_samplers.resize(LIGHT_TYPE_COUNT);
  std::unique_ptr<LightSampler> sampler;

  sampler = std::make_unique<AreaLightSampler>(m_context);
  m_program["SampleAreaLights"]->set(sampler.get()->GetProgram());
  m_samplers[LIGHT_TYPE_AREA] = std::move(sampler);

  sampler = std::make_unique<DistantLightSampler>(m_context);
  m_program["SampleDistantLights"]->set(sampler.get()->GetProgram());
  m_samplers[LIGHT_TYPE_DISTANT] = std::move(sampler);

  sampler = std::make_unique<PointLightSampler>(m_context);
  m_program["SamplePointLights"]->set(sampler.get()->GetProgram());
  m_samplers[LIGHT_TYPE_POINT] = std::move(sampler);
}

} // namespace torch