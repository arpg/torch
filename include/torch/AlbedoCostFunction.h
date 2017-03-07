#pragma once

#include <vector>
#include <lynx/lynx.h>
#include <torch/Core.h>

namespace torch
{

class AlbedoCostFunction : public lynx::CostFunction
{
  public:

    AlbedoCostFunction(std::shared_ptr<MatteMaterial> material,
        std::shared_ptr<Mesh> mesh);

    virtual ~AlbedoCostFunction();

    void AddKeyframe(std::shared_ptr<Keyframe> keyframe);

    lynx::Matrix* CreateJacobianMatrix() override;

    void Evaluate(float const* const* parameters, float* residuals) override;

    void Evaluate(size_t offset, size_t size, float const* const* parameters,
        float* residuals, lynx::Matrix* jacobian) override;

    void Evaluate(float const* const* parameters, float* residuals,
        float* gradient) override;

    void ClearJacobian();

  protected:

    void PrepareEvaluation();

    void CreatePixelVertexBuffer();

    void ComputeJacobian();

    void CreateReferenceBuffer();

  private:

    void Initialize();

    void SetDimensions();

    void CreateJacobianBuffer();

    void CreateBoundingBoxBuffer();

    void CreateAdjacencyBuffers();

    void CreateCaptureProgram();

    void CreateBoundsProgram();

    void CreateKeyframeSet();

  protected:

    bool m_locked;

    std::shared_ptr<MatteMaterial> m_material;

    std::shared_ptr<Mesh> m_mesh;

    std::unique_ptr<KeyframeSet> m_keyframes;

    std::shared_ptr<SparseMatrix> m_jacobian;

    optix::Buffer m_boundingBoxes;

    optix::Buffer m_neighborOffsets;

    optix::Buffer m_neighborIndices;

    optix::Program m_captureProgram;

    unsigned int m_captureProgramId;

    optix::Program m_boundsProgram;

    unsigned int m_boundsProgramId;

    float* m_jacobianValues;

    float* m_referenceValues;

    unsigned int m_valueCount;

    unsigned int* m_rowIndices;

    unsigned int* m_colIndices;

    bool m_dirty;
};

} // namespace torch