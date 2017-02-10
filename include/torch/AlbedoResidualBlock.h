#pragma once

#include <torch/Core.h>

namespace torch
{

class AlbedoResidualBlock
{
  public:

    AlbedoResidualBlock(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<ReferenceImage> reference);

    optix::Program GetAddProgram() const;

    std::shared_ptr<SparseMatrix> GetJacobian() const;

  private:

    void Initialize();

    void CreateAdjacencyMap();

    void CreateBoundingBoxProgram();

    void CreateBoundingBoxBuffer();

    void GetBoundingBoxes(std::vector<uint4>& bboxes);

    void CreateJacobian(const std::vector<uint4>& bboxes);

  protected:

    std::shared_ptr<Mesh> m_mesh;

    std::shared_ptr<ReferenceImage> m_reference;

    std::shared_ptr<SparseMatrix> m_jacobian;

    optix::Buffer m_neighborOffsets;

    optix::Buffer m_neighborIndices;

    optix::Buffer m_bboxBuffer;

    optix::Program m_bboxProgram;

    unsigned int m_bboxProgramId;
};

} // namespace torch