#pragma once

#include <lynx/lynx.h>
#include <torch/Core.h>

namespace torch
{

class DarkenCostFunction : public lynx::CostFunction
{
  public:

    DarkenCostFunction(size_t size);

    virtual ~DarkenCostFunction();

    float GetWeight() const;

    void SetWeight(float weight);

    void SetValues(float* values);

    lynx::Matrix* CreateJacobianMatrix() override;

    void Evaluate(float const* const* parameters, float* residuals) override;

    void Evaluate(size_t offset, size_t size, float const* const* parameters,
        float* residuals, lynx::Matrix* jacobian) override;

  private:

    void Initialize();

    void SetDimensions();

    void AllocateValues();

  protected:

    size_t m_size;

    float m_weight;

    float* m_values;
};

} // namespace torch