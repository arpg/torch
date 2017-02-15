#include <torch/ReflectanceCostFunction.h>
#include <torch/MatteMaterial.h>
#include <torch/device/ReflectanceCostFunction.cuh>

#include <iostream>

namespace torch
{

ReflectanceCostFunction::ReflectanceCostFunction(
    std::shared_ptr<MatteMaterial> material) :
  m_material(material)
{
  Initialize();
}

ReflectanceCostFunction::~ReflectanceCostFunction()
{
}

lynx::Matrix* ReflectanceCostFunction::CreateJacobianMatrix()
{
  const size_t count = GetResidualCount();
  return new lynx::DiagonalMatrix(count);
}

void ReflectanceCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  const size_t count = GetResidualCount();
  torch::Evaluate(parameters[0], residuals, nullptr, count);
}

void ReflectanceCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  lynx::DiagonalMatrix* matrix;
  matrix = dynamic_cast<lynx::DiagonalMatrix*>(jacobian);
  LYNX_ASSERT(matrix, "expected jacobian to be BlockDiagonalMatrix type");

  const float* params = parameters[0];
  float* J = &matrix->GetValues()[offset];
  float* r = &residuals[offset];

  torch::Evaluate(params, r, J, size);
}

void ReflectanceCostFunction::Initialize()
{
  const size_t count = 3 * m_material->GetAlbedoCount();
  lynx::CostFunction::m_residualCount = count;
  lynx::CostFunction::m_parameterBlockSizes.push_back(count);
  lynx::CostFunction::m_maxEvaluationBlockSize = count;
}

} // namespace torch