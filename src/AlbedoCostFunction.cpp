#include <torch/AlbedoCostFunction.h>
#include <torch/Context.h>
#include <torch/MatteMaterial.h>
#include <torch/Keyframe.h>

#include <iostream>

namespace torch
{

AlbedoCostFunction::AlbedoCostFunction(
    std::shared_ptr<MatteMaterial> material) :
  m_dirty(true),
  m_locked(false),
  m_material(material),
  m_jacobianValues(nullptr)
{
  Initialize();
}

AlbedoCostFunction::~AlbedoCostFunction()
{
}

void AlbedoCostFunction::AddKeyframe(std::shared_ptr<Keyframe> keyframe)
{
  LYNX_ASSERT(!m_locked, "new keyframes cannot be added");
  const size_t residualCount = 3 * keyframe->GetValidPixelCount();
  lynx::CostFunction::m_residualCount += residualCount;
  lynx::CostFunction::m_maxEvaluationBlockSize += residualCount;
  m_keyframes.push_back(keyframe);
  m_dirty = true;
}

lynx::Matrix* AlbedoCostFunction::CreateJacobianMatrix()
{
  LYNX_ASSERT(!m_dirty, "cost function is out-of-data");

  m_locked = true;
  const size_t rows = GetResidualCount();
  const size_t cols = GetParameterCount();
  return new lynx::Matrix3C(m_jacobianValues, rows, cols);
}

void AlbedoCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  LYNX_ASSERT(!m_dirty, "cost function is out-of-data");

  const size_t rows = GetResidualCount();
  const size_t cols = GetParameterCount();
  lynx::Matrix3C jacobian(m_jacobianValues, rows, cols);
  jacobian.RightMultiply(parameters[0], residuals);
  lynx::Sub(m_referenceValues, residuals, residuals, GetResidualCount());
}

void AlbedoCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  LYNX_ASSERT(!m_dirty, "cost function is out-of-data");
  LYNX_ASSERT(jacobian->GetValues() == m_jacobianValues, "invalid jacobian");
  LYNX_ASSERT(offset == 0, "sub-problem cannot be evaluated");
  LYNX_ASSERT(size == GetResidualCount(), "sub-problem cannot be evaluated");

  jacobian->RightMultiply(parameters[0], residuals);
  lynx::Sub(m_referenceValues, residuals, residuals, GetResidualCount());
}

void AlbedoCostFunction::UpdateCostFunction()
{
  ComputeJacobian();
  CreateReferenceBuffer();
  m_dirty = false;
}

void AlbedoCostFunction::ComputeJacobian()
{
  LYNX_ASSERT(!m_keyframes.empty(), "no keyframes have been added");

  const size_t rows = GetResidualCount();
  const size_t cols = GetParameterCount();
  m_jacobian->setSize(cols, rows);

  // set jacobian to zero
  // update material buffer
  // set "computeDerivs" flag
  // trace scene
  // unset "computeDerivs" flag

  CUdeviceptr pointer = m_jacobian->getDevicePointer(0);
  m_jacobianValues = reinterpret_cast<float*>(pointer);

  throw std::string("function not implemented: ") + __FUNCTION__;
}

void AlbedoCostFunction::CreateReferenceBuffer()
{
  throw std::string("function not implemented: ") + __FUNCTION__;
}

void AlbedoCostFunction::Initialize()
{
  SetDimensions();
  CreateBuffer();
}

void AlbedoCostFunction::SetDimensions()
{
  const size_t paramCount = 3 * m_material->GetAlbedoCount();
  lynx::CostFunction::m_parameterBlockSizes.push_back(paramCount);
  lynx::CostFunction::m_maxEvaluationBlockSize = 0;
  lynx::CostFunction::m_residualCount = 0;
}

void AlbedoCostFunction::CreateBuffer()
{
  std::shared_ptr<Context> context;
  context = m_material->GetContext();
  m_jacobian = context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_jacobian->setFormat(RT_FORMAT_FLOAT3);
  m_jacobian->setSize(0u, 0u);
}

} // namespace torch