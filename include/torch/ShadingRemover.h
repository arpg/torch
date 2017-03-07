#pragma once

#include <torch/Core.h>

namespace torch
{

class ShadingRemover
{
  public:

    ShadingRemover(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<MatteMaterial> material);

    virtual ~ShadingRemover();

    void SetSampleCount(unsigned int count);

    void Remove();

  private:

    void Initialize();

    void CreateDummyMaterial();

    void CreateDummyGeometry();

    void CreateProgram();

  protected:

    std::shared_ptr<Mesh> m_mesh;

    std::shared_ptr<MatteMaterial> m_material;

    unsigned int m_sampleCount;

    optix::Program m_closestHitProgram;

    optix::Program m_program;

    unsigned int m_programId;

    optix::Material m_dummyMaterial;

    optix::Geometry m_dummyGeometry;

    optix::GeometryGroup m_dummyGroup;

    optix::GeometryInstance m_dummyInstance;

    optix::Acceleration m_dummyAccel;
};

} // namespace torch