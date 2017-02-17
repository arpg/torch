#pragma once

#include <vector>
#include <lynx/lynx.h>
#include <torch/Core.h>

namespace torch
{

class AlbedoCostFunction : public lynx::CostFunction
{
  public:

    AlbedoCostFunction(std::shared_ptr<MatteMaterial> material);

    virtual ~AlbedoCostFunction();

    void AddKeyframe(std::shared_ptr<Keyframe> keyframe);

    lynx::Matrix* CreateJacobianMatrix() override;

    void Evaluate(float const* const* parameters, float* residuals) override;

    void Evaluate(size_t offset, size_t size, float const* const* parameters,
        float* residuals, lynx::Matrix* jacobian) override;

    void ClearJacobian();

  protected:

    void PrepareEvaluation();

    void ComputeJacobian();

    void ResetJacobian();

    void CreateReferenceBuffer();

  private:

    void Initialize();

    void SetDimensions();

    void CreateBuffer();

    void CreateProgram();

    void CreateKeyframeSet();

  protected:

    bool m_locked;

    std::shared_ptr<MatteMaterial> m_material;

    std::unique_ptr<KeyframeSet> m_keyframes;

    optix::Buffer m_jacobian;

    optix::Program m_program;

    unsigned int m_programId;

    float* m_jacobianValues;

    float* m_referenceValues;
};

} // namespace torch