#pragma once

#include <torch/LightSampler.h>

namespace torch
{

class EnvironmentLightSampler : public LightSampler
{
  public:

    EnvironmentLightSampler(std::shared_ptr<Context> context);

    optix::Program GetProgram() const override;

    float GetLuminance() const override;

    void Clear() override;

    void Update() override;

    void Add(const EnvironmentLightData& light);

  protected:

    void UpdateDistribution();

    void UpdateBuffers();

    template <typename T>
    static void WriteBuffer(optix::Buffer buffer, const std::vector<T>& data);

  private:

    void Initialize();

    void CreateProgram();

    void CreateDistribution();

    void CreateBuffers();

    void CreateRowProgramsBuffer();

    void CreateColProgramsBuffer();

    void CreateRowOffsetsBuffer();

    void CreateColOffsetsBuffer();

    void CreateRotationsBuffer();

    void CreateRadianceBuffer();

  protected:

    optix::Program m_program;

    std::vector<EnvironmentLightData> m_lights;

    std::unique_ptr<Distribution> m_distribution;

    optix::Buffer m_rowPrograms;

    optix::Buffer m_colPrograms;

    optix::Buffer m_rowOffsets;

    optix::Buffer m_colOffsets;

    optix::Buffer m_rotations;

    optix::Buffer m_radiance;
};

} // namespace torch