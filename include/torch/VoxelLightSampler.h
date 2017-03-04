#pragma once

#include <torch/LightSampler.h>

namespace torch
{

class VoxelLightSampler : public LightSampler
{
  public:

    VoxelLightSampler(std::shared_ptr<Context> context);

    virtual ~VoxelLightSampler();

    optix::Program GetProgram() const override;

    float GetLuminance() const override;

    void Add(const VoxelLightData& light);

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

    void CreateSubDataBuffer();

    void CreateSampleProgramsBuffer();

    void CreateRadianceBuffer();

    void CreateDerivativeBuffer();

    void CreateEmptyDerivativeBuffer();

  protected:

    optix::Program m_program;

    optix::Buffer m_subdata;

    optix::Buffer m_radiance;

    optix::Buffer m_derivatives;

    optix::Buffer m_emptyDerivs;

    optix::Buffer m_samplePrograms;

    std::vector<VoxelLightData> m_lights;

    std::unique_ptr<Distribution1D> m_distribution;
};

} // namespace torch