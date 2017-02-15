#pragma once

#include <vector>
#include <lynx/lynx.h>
#include <torch/Core.h>

namespace torch
{

class LightCostFunction : public lynx::CostFunction
{
  public:

    LightCostFunction(std::shared_ptr<EnvironmentLight> light);

    virtual ~LightCostFunction();

    void AddKeyframe(std::shared_ptr<Keyframe> keyframe);

    lynx::Matrix* CreateJacobianMatrix() const override;

    void Evaluate(float const* const* parameters, float* residuals) override;

    void Evaluate(size_t offset, size_t size, float const* const* parameters,
        float* residuals, lynx::Matrix* jacobian) override;

  protected:

    void PrepareEvaluation();

    void ComputeJacobian();

  private:

    void Initialize();

  protected:

    bool m_dirty;

    std::shared_ptr<EnvironmentLight> m_light;

    std::vector<std::shared_ptr<Keyframe>> m_keyframes;
};

} // namespace torch