#include <torch/VoxelActivationCostFunction.h>
#include <torch/Exception.h>
#include <torch/VoxelLight.h>
#include <torch/device/VoxelActivationCostFunction.cuh>

namespace torch
{

VoxelActivationCostFunction::VoxelActivationCostFunction(
    std::shared_ptr<VoxelLight> light) :
  m_light(light),
  m_bias(1.0f),
  m_innerScale(1.0f),
  m_outerScale(1.0f)
{
  Initialize();
}

VoxelActivationCostFunction::~VoxelActivationCostFunction()
{
}

float VoxelActivationCostFunction::GetBias() const
{
  return m_bias;
}

void VoxelActivationCostFunction::SetBias(float bias)
{
  m_bias = bias;
}

float VoxelActivationCostFunction::GetInnerScale() const
{
  return m_innerScale;
}

void VoxelActivationCostFunction::SetInnerScale(float scale)
{
  m_innerScale = scale;
}

float VoxelActivationCostFunction::GetOuterScale() const
{
  return m_outerScale;
}

void VoxelActivationCostFunction::SetOuterScale(float scale)
{
  m_outerScale = scale;
}

lynx::Matrix* VoxelActivationCostFunction::CreateJacobianMatrix()
{
  const size_t count = m_light->GetVoxelCount();
  return new lynx::BlockDiagonalMatrix(1, 3, count);
}

void VoxelActivationCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  const size_t count = m_light->GetVoxelCount();

  torch::Evaluate(parameters[0], residuals, nullptr, count,
      m_bias, m_innerScale, m_outerScale);
}

void VoxelActivationCostFunction::Evaluate(size_t offset, size_t size,
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

void VoxelActivationCostFunction::Evaluate(const float* const* parameters,
    float* residuals, float* gradient)
{
  TORCH_THROW("not implemented");
}

void VoxelActivationCostFunction::Initialize()
{
  const size_t count = m_light->GetVoxelCount();
  lynx::CostFunction::m_residualCount = count;
  lynx::CostFunction::m_parameterBlockSizes.push_back(3 * count);
  lynx::CostFunction::m_maxEvaluationBlockSize = count;
}

} // namespace torch