#pragma once

#include <torch/GeometrySampler.h>

namespace torch
{

class GeometryGroupSampler : public GeometrySampler
{
  public:

    GeometryGroupSampler(std::shared_ptr<Context> context);

    optix::Program GetProgram() const override;

    void Add(const GeometryGroupData& group);

    void Clear() override;

    void Update() override;

  private:

    void Initialize();

    void CreateProgram();

    void CreateBuffer();

  protected:

    optix::Buffer m_buffer;

    optix::Program m_program;

    std::vector<GeometryGroupData> m_groups;
};

} // namespace torch