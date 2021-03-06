#pragma once

#include <torch/GeometrySampler.h>

namespace torch
{

class SphereSampler : public GeometrySampler
{
  public:

    SphereSampler(std::shared_ptr<Context> context);

    optix::Program GetProgram() const override;

    void Add(const SphereData& sphere);

    void Clear() override;

    void Update() override;

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