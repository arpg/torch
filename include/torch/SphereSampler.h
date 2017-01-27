#pragma once

#include <vector>
#include <torch/GeometryData.h>
#include <torch/GeometrySampler.h>

namespace torch
{

class SphereSampler : public GeometrySampler
{
  public:

    SphereSampler(std::shared_ptr<Context> context);

    optix::Program GetProgram() const override;

    void Clear() override;

    void Update() override;

    unsigned int Add(const SphereData& sphere);

  private:

    void Initialize();

    void CreateProgram();

    void CreateBuffer();

  protected:

    optix::Buffer m_buffer;

    optix::Program m_program;

    std::vector<SphereData> m_spheres;
};

} // namespace torch