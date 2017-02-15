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

    void UpdateBuffers();

    template <typename T>
    static void WriteBuffer(optix::Buffer buffer, const std::vector<T>& data);

  private:

    void Initialize();

    void CreateProgram();

    void CreateDistribution();

    void CreateRotationsBuffer();

    void CreateSampleProgramsBuffer();

    void CreateOffsetBuffer();

    void CreateRadianceBuffer();

    void CreateDerivativeBuffer();

    void CreateEmptyDerivativeBuffer();

  protected:

    optix::Program m_program;

    optix::Buffer m_rotations;

    optix::Buffer m_samplePrograms;

    optix::Buffer m_offsets;

    optix::Buffer m_radiance;

    optix::Buffer m_derivatives;

    optix::Buffer m_emptyDerivs;

    std::vector<EnvironmentLightData> m_lights;

    std::unique_ptr<Distribution1D> m_distribution;
};

} // namespace torch