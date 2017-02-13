#pragma once

#include <lynx/CostFunction.h>
#include <torch/Core.h>

namespace torch
{

class ActivationCostFunction : public lynx::CostFunction
{
  public:

    ActivationCostFunction(std::shared_ptr<EnvironmentLight> light);

    virtual ~ActivationCostFunction();

    lynx::Matrix* CreateJacobianMatrix() const override;

    void Evaluate(float const* const* parameters, float* residuals) override;

    void Evaluate(size_t block, float const* const* parameters,
        float* residuals, lynx::Matrix* jacobian) override;

  private:

    void Initialize();

  protected:

    std::shared_ptr<EnvironmentLight> m_light;
};

}