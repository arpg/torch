#pragma once

#include <lynx/lynx.h>
#include <torch/Core.h>

namespace torch
{

class ActivationCostFunction : public lynx::CostFunction
{
  public:

    ActivationCostFunction(std::shared_ptr<EnvironmentLight> light);

    virtual ~ActivationCostFunction();

    float GetBias() const;

    void SetBias(float bias);

    float GetInnerScale() const;

    void SetInnerScale(float scale);

    float GetOuterScale() const;

    void SetOuterScale(float scale);

    lynx::Matrix* CreateJacobianMatrix() override;

    void Evaluate(float const* const* parameters, float* residuals) override;

    void Evaluate(size_t offset, size_t size, float const* const* parameters,
        float* residuals, lynx::Matrix* jacobian) override;

    void Evaluate(float const* const* parameters, float* residuals,
        float* gradient) override;

  private:

    void Initialize();

  protected:

    std::shared_ptr<EnvironmentLight> m_light;

    float m_bias;

    float m_innerScale;

    float m_outerScale;

    std::unique_ptr<lynx::BlockDiagonalMatrix> m_jacobian;
};

} // namespace torch