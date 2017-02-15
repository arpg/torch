#include <torch/LightCostFunction.h>
#include <torch/device/LightCostFunction.cuh>
#include <torch/EnvironmentLight.h>
#include <torch/Keyframe.h>

#include <iostream>

namespace torch
{

LightCostFunction::LightCostFunction(
    std::shared_ptr<EnvironmentLight> light) :
  m_dirty(true),
  m_light(light)
{
  Initialize();
}

LightCostFunction::~LightCostFunction()
{
}

void LightCostFunction::AddKeyframe(std::shared_ptr<Keyframe> keyframe)
{
  const size_t residualCount = 3 * keyframe->GetValidPixelCount();
  lynx::CostFunction::m_residualCount += residualCount;
  lynx::CostFunction::m_maxEvaluationBlockSize += residualCount;
  m_keyframes.push_back(keyframe);
  m_dirty = true;
}

lynx::Matrix* LightCostFunction::CreateJacobianMatrix() const
{
  // const size_t count = m_light->GetDirectionCount();
  // return new lynx::BlockDiagonalMatrix(1, 3, count);
  throw std::string("function not implemented: ") + __FUNCTION__;
}

void LightCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  PrepareEvaluation();

  throw std::string("function not implemented: ") + __FUNCTION__;
}

void LightCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  PrepareEvaluation();

  // cast matrix to dense matrix
  // check if cast was successful

  // copy member matrix to argument matrix

  Evaluate(parameters, residuals);

  throw std::string("function not implemented: ") + __FUNCTION__;
}

void LightCostFunction::PrepareEvaluation()
{
  if (m_dirty)
  {
    ComputeJacobian();
    m_dirty = false;
  }
}

void LightCostFunction::ComputeJacobian()
{
  throw std::string("function not implemented: ") + __FUNCTION__;
}

void LightCostFunction::Initialize()
{
  const size_t paramCount = 3 * m_light->GetDirectionCount();
  lynx::CostFunction::m_parameterBlockSizes.push_back(paramCount);
  lynx::CostFunction::m_maxEvaluationBlockSize = 0;
  lynx::CostFunction::m_residualCount = 0;
}

} // namespace torch