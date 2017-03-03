#include <torch/DarkenCostFunction.h>
#include <torch/EnvironmentLight.h>
#include <torch/device/DarkenCostFunction.cuh>

namespace torch
{

DarkenCostFunction::DarkenCostFunction(size_t size) :
  m_size(size),
  m_weight(1.0f),
  m_values(nullptr)
{
  Initialize();
}

DarkenCostFunction::~DarkenCostFunction()
{
  cudaFree(m_values);
}

float DarkenCostFunction::GetWeight() const
{
  return m_weight;
}

void DarkenCostFunction::SetWeight(float weight)
{
  m_weight = weight;
}

void DarkenCostFunction::SetValues(float* values)
{
  lynx::Copy(m_values, values, m_size);
}

lynx::Matrix* DarkenCostFunction::CreateJacobianMatrix()
{
  return new lynx::DiagonalMatrix(m_size);
}

void DarkenCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  torch::Evaluate(parameters[0], residuals, nullptr, m_values,
      m_weight, m_size);
}

void DarkenCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  lynx::DiagonalMatrix* matrix;
  matrix = dynamic_cast<lynx::DiagonalMatrix*>(jacobian);
  LYNX_ASSERT(matrix, "expected jacobian to be DiagonalMatrix type");

  const float* params = parameters[0];
  float* J = &matrix->GetValues()[3 * offset];
  float* r = &residuals[offset];

  torch::Evaluate(params, r, J, m_values, m_weight, size);
}

void DarkenCostFunction::Initialize()
{
  SetDimensions();
  AllocateValues();
}

void DarkenCostFunction::SetDimensions()
{
  lynx::CostFunction::m_residualCount = m_size;
  lynx::CostFunction::m_parameterBlockSizes.push_back(m_size);
  lynx::CostFunction::m_maxEvaluationBlockSize = m_size;
}

void DarkenCostFunction::AllocateValues()
{
  LYNX_CHECK_CUDA(cudaMalloc(&m_values, sizeof(float) * m_size));
}

} // namespace torch