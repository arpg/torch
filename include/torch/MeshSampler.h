#pragma once

#include <torch/GeometrySampler.h>

namespace torch
{

class MeshSampler : public GeometrySampler
{
  public:

    MeshSampler(std::shared_ptr<Context> context);

    optix::Program GetProgram() const override;

    void Add(const MeshData& mesh);

    void Clear() override;

    void Update() override;

  private:

    void Initialize();

    void CreateProgram();

    void CreateBuffer();

  protected:

    optix::Buffer m_buffer;

    optix::Program m_program;

    std::vector<MeshData> m_meshes;
};

} // namespace torch