#pragma once

#include <vector>
#include <torch/Distribution.h>
#include <torch/LightData.h>
#include <torch/LightSampler.h>

namespace torch
{

class PointLightSampler : public LightSampler
{
  public:

    PointLightSampler(std::shared_ptr<Context> context);

    optix::Program GetProgram() const override;

    float GetLuminance() const override;

    void Clear() override;

    void Update() override;

    void Add(const PointLightData& light);

  protected:

    void UpdateDistribution();

    void UpdateLightBuffer();

  private:

    void Initialize();

    void CreateDistribution();

    void CreateLightBuffer();

    void CreateProgram();

  protected:

    optix::Buffer m_buffer;

    optix::Program m_program;

    std::vector<PointLightData> m_lights;

    std::unique_ptr<Distribution> m_distribution;
};

} // namespace torch