#pragma once

#include <torch/Light.h>
#include <torch/Spectrum.h>

namespace torch
{

class VoxelLight : public Light
{
  public:

    VoxelLight(std::shared_ptr<Context> context);

    virtual ~VoxelLight();

    unsigned int GetVoxelCount() const;

    void SetRadiance(const Spectrum& radiance);

    void SetRadiance(float r, float g, float b);

    void SetRadiance(size_t index, const Spectrum& radiance);

    void SetRadiance(const std::vector<Spectrum>& radiance);

    uint3 GetDimensions() const;

    void SetDimensions(uint x, uint y, uint z);

    void SetDimensions(uint dims);

    void SetVoxelSize(float size);

    void SetDerivativeBuffer(optix::Buffer buffer);

    void BuildScene(Link& link) override;

    optix::Buffer GetRadianceBuffer() const;

  protected:

    void UpdateDistribution();

    void UpdateRadianceBuffer();

    void UpdateSampler(Link& link);

  private:

    void Initialize();

    void CreateDistribution();

    void CreateRadianceBuffer();

  protected:

    float m_voxelSize;

    uint3 m_gridDimensions;

    std::vector<Spectrum> m_radiance;

    std::unique_ptr<Distribution1D> m_distribution;

    optix::Buffer m_radianceBuffer;

    optix::Buffer m_derivBuffer;
};

} // namespace torch