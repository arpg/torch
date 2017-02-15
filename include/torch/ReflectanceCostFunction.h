#pragma once

#include <lynx/lynx.h>
#include <torch/Core.h>

namespace torch
{

class ReflectanceCostFunction : public lynx::CostFunction
{
  public:

    ReflectanceCostFunction(std::shared_ptr<MatteMaterial> material);

    virtual ~ReflectanceCostFunction();

    lynx::Matrix* CreateJacobianMatrix() override;

    void Evaluate(float const* const* parameters, float* residuals) override;

    void Evaluate(size_t offset, size_t size, float const* const* parameters,
        float* residuals, lynx::Matrix* jacobian) override;

  private:

    void Initialize();

  protected:

    std::shared_ptr<MatteMaterial> m_material;
};

} // namespace torch