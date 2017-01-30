#pragma once

#include <torch/LightSampler.h>

namespace torch
{

class EnvironmentLightSampler : public LightSampler
{
  public:

    EnvironmentLightSampler(std::shared_ptr<Context> context);

    ~EnvironmentLightSampler();

    optix::Program GetProgram() const override;

    float GetLuminance() const override;

    void Add(const EnvironmentLightData& light);

    void Clear() override;

    void Update() override;

  protected:

    void UpdateDistribution();

    void UpdateLightDistributions();

    void UpdateBuffers();

    template <typename T>
    static void WriteBuffer(optix::Buffer buffer, const std::vector<T>& data);

  private:

    void Initialize();

    void CreateProgram();

    void CreateDistribution();

    void CreateBuffers();

    void CreateSampleProgramsBuffer();

    void CreateLightOffsetsBuffer();

    void CreateRowOffsetsBuffer();

    void CreateRotationsBuffer();

    void CreateRadianceBuffer();

  protected:

    optix::Program m_program;

    optix::Buffer m_samplePrograms;

    optix::Buffer m_lightOffsets;

    optix::Buffer m_rowOffsets;

    optix::Buffer m_rotations;

    optix::Buffer m_radiance;

    std::vector<EnvironmentLightData> m_lights;

    std::unique_ptr<Distribution1D> m_distribution;

    std::vector<std::unique_ptr<Distribution2D>> m_lightDistributions;
};

} // namespace torch