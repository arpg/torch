#pragma once

#include <vector>
#include <lynx/lynx.h>
#include <torch/Core.h>

namespace torch
{

class VoxelCostFunction : public lynx::CostFunction
{
  public:

    VoxelCostFunction(std::shared_ptr<VoxelLight> light);

    virtual ~VoxelCostFunction();

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

    void CreateReferenceBuffer();

    void ComputeJacobian();

    void ResetJacobian();

  private:

    void Initialize();

    void SetDimensions();

    void CreateBuffer();

    void CreateProgram();

    void CreateKeyframeSet();

  protected:

    bool m_locked;

    std::shared_ptr<VoxelLight> m_light;

    std::unique_ptr<KeyframeSet> m_keyframes;

    optix::Buffer m_jacobian;

    optix::Program m_program;

    unsigned int m_programId;

    float* m_jacobianValues;

    float* m_referenceValues;

    unsigned int m_iterations;
};

} // namespace torch