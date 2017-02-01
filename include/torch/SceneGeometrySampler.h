#pragma once

#include <torch/Core.h>

namespace torch
{

class SceneGeometrySampler
{
  public:

    SceneGeometrySampler(std::shared_ptr<Context> context);

    optix::Program GetProgram() const;

    void Add(const GeometryGroupData& group);

    void Add(const MeshData& mesh);

    void Add(const SphereData& sphere);

    void Clear();

    void Update();

  private:

    void Initialize();

    void CreateProgram();

    void CreateSamplers();

  protected:

    std::shared_ptr<Context> m_context;

    optix::Program m_program;

    std::vector<std::unique_ptr<GeometrySampler>> m_samplers;
};

} // namespace torch