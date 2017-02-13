#include <torch/ActivationCostFunction.h>
#include <torch/EnvironmentLight.h>

namespace torch
{

ActivationCostFunction::ActivationCostFunction(
    std::shared_ptr<EnvironmentLight> light) :
  m_light(light)
{
  Initialize();
}

ActivationCostFunction::~ActivationCostFunction()
{
}

lynx::Matrix* ActivationCostFunction::CreateJacobianMatrix() const
{
  return nullptr;
}

void ActivationCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{

}

void ActivationCostFunction::Evaluate(size_t block,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
}

void ActivationCostFunction::Initialize()
{
  const size_t count = 3 * m_light->GetDirectionCount();
  lynx::CostFunction::m_residualCount = count;
  lynx::CostFunction::m_parameterBlockSizes.push_back(count);
  lynx::CostFunction::m_evaluationBlockSizes.push_back(count);
}

} // namespace torch