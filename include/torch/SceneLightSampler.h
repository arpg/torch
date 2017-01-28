#pragma once

#include <torch/Core.h>

namespace torch
{

class SceneLightSampler
{
  public:

    SceneLightSampler(std::shared_ptr<Context> context);

    optix::Program GetProgram() const;

    void Add(const AreaLightData& light);

    void Add(const DistantLightData& light);

    void Add(const PointLightData& light);

    void Clear();

    void Update();

  protected:

    void UpdateLightSamplers();

    void UpdateDistribution();

  private:

    void Initialize();

    void CreateProgram();

    void CreateDistribution();

    void CreateLightSamplers();

  protected:

    std::shared_ptr<Context> m_context;

    optix::Program m_program;

    std::unique_ptr<Distribution> m_distribution;

    std::vector<std::unique_ptr<LightSampler>> m_samplers;
};

} // namespace torch