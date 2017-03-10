#pragma once

#include <torch/Core.h>

namespace torch
{

class AlbedoBaker
{
  public:

    AlbedoBaker(std::shared_ptr<Scene> scene);

    unsigned int GetSampleCount() const;

    void SetSampleCount(unsigned int count);

    void Bake(std::shared_ptr<MatteMaterial> material,
        std::shared_ptr<Mesh> mesh);

  private:

    void Initialize();

    void CreateMaterial();

    void CreateGeometry();

    void CreateScratchBuffer();

    void CreateBakeProgram();

    void CreateCopyProgram();

  protected:

    std::shared_ptr<Scene> m_scene;

    optix::Buffer m_scratchBuffer;

    optix::Program m_bakeProgram;

    unsigned int m_bakeProgramId;

    optix::Program m_copyProgram;

    unsigned int m_copyProgramId;

    unsigned int m_sampleCount;

    optix::Material m_material;

    optix::Program m_hitProgram;

    optix::Geometry m_geometry;

    optix::GeometryGroup m_group;

    optix::GeometryInstance m_instance;

    optix::Acceleration m_accel;
};

} // namespace torch