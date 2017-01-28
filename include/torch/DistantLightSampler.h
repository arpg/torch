#pragma once

#include <vector>
#include <torch/Distribution.h>
#include <torch/LightData.h>
#include <torch/LightSampler.h>

namespace torch
{

class DistantLightSampler : public LightSampler
{
  public:

    DistantLightSampler(std::shared_ptr<Context> context);

    optix::Program GetProgram() const override;

    float GetLuminance() const override;

    void Clear() override;

    void Update() override;

    void Add(const DistantLightData& light);

  protected:

    void UpdateDistribution();

    void UpdateLightBuffer();

  private:

    void Initialize();

    void CreateProgram();

    void CreateDistribution();

    void CreateLightBuffer();

  protected:

    optix::Buffer m_buffer;

    optix::Program m_program;

    std::vector<DistantLightData> m_lights;

    std::unique_ptr<Distribution> m_distribution;
};

} // namespace torch