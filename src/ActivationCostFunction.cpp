#include <torch/ActivationCostFunction.h>
#include <torch/EnvironmentLight.h>
#include <torch/device/ActivationCostFunction.cuh>

namespace torch
{

ActivationCostFunction::ActivationCostFunction(
    std::shared_ptr<EnvironmentLight> light) :
  m_light(light),
  m_bias(1.0f),
  m_innerScale(1.0f),
  m_outerScale(1.0f)
{
  Initialize();
}

ActivationCostFunction::~ActivationCostFunction()
{
}

float ActivationCostFunction::GetBias() const
{
  return m_bias;
}

void ActivationCostFunction::SetBias(float bias)
{
  m_bias = bias;
}

float ActivationCostFunction::GetInnerScale() const
{
  return m_innerScale;
}

void ActivationCostFunction::SetInnerScale(float scale)
{
  m_innerScale = scale;
}

float ActivationCostFunction::GetOuterScale() const
{
  return m_outerScale;
}

void ActivationCostFunction::SetOuterScale(float scale)
{
  m_outerScale = scale;
}

lynx::Matrix* ActivationCostFunction::CreateJacobianMatrix()
{
  const size_t count = m_light->GetDirectionCount();
  return new lynx::BlockDiagonalMatrix(1, 3, count);
}

void ActivationCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  const size_t count = m_light->GetDirectionCount();

  torch::Evaluate(parameters[0], residuals, nullptr, count,
      m_bias, m_innerScale, m_outerScale);
}

void ActivationCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  lynx::BlockDiagonalMatrix* matrix;
  matrix = dynamic_cast<lynx::BlockDiagonalMatrix*>(jacobian);
  LYNX_ASSERT(matrix, "expected jacobian to be BlockDiagonalMatrix type");

  const float* params = parameters[0];
  float* J = &matrix->GetValues()[3 * offset];
  float* r = &residuals[offset];

  torch::Evaluate(params, r, J, size, m_bias, m_innerScale, m_outerScale);
}

void ActivationCostFunction::Initialize()
{
  const size_t count = m_light->GetDirectionCount();
  lynx::CostFunction::m_residualCount = count;
  lynx::CostFunction::m_parameterBlockSizes.push_back(3 * count);
  lynx::CostFunction::m_maxEvaluationBlockSize = count;
}

} // namespace torch