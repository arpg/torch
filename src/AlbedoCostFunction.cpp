#include <torch/AlbedoCostFunction.h>
#include <torch/Context.h>
#include <torch/MatteMaterial.h>
#include <torch/Keyframe.h>
#include <torch/KeyframeSet.h>
#include <torch/PtxUtil.h>

#include <iostream>

namespace torch
{

AlbedoCostFunction::AlbedoCostFunction(
    std::shared_ptr<MatteMaterial> material) :
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
  m_referenceValues = nullptr;
  m_jacobianValues = nullptr;
  m_keyframes->Add(keyframe);
}

lynx::Matrix* AlbedoCostFunction::CreateJacobianMatrix()
{
  m_locked = true;
  PrepareEvaluation();
  const size_t rows = GetResidualCount(); // TODO: check size
  const size_t cols = GetParameterCount(); // TODO: check size
  return new lynx::Matrix3C(m_jacobianValues, rows, cols); // TODO: change type
}

void AlbedoCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  PrepareEvaluation();
  const size_t rows = GetResidualCount(); // TODO: check size
  const size_t cols = GetParameterCount(); // TODO: check size
  lynx::Matrix3C jacobian(m_jacobianValues, rows, cols); // TODO: change type
  jacobian.RightMultiply(parameters[0], residuals);
  lynx::Sub(m_referenceValues, residuals, residuals, GetResidualCount());
}

void AlbedoCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  LYNX_ASSERT(jacobian->GetValues() == m_jacobianValues, "invalid jacobian");
  LYNX_ASSERT(offset == 0, "sub-problem cannot be evaluated");
  LYNX_ASSERT(size == GetResidualCount(), "sub-problem cannot be evaluated");

  PrepareEvaluation();
  jacobian->RightMultiply(parameters[0], residuals);
  lynx::Sub(m_referenceValues, residuals, residuals, GetResidualCount());
}

void AlbedoCostFunction::ClearJacobian()
{
  m_jacobianValues = nullptr;
}

void AlbedoCostFunction::PrepareEvaluation()
{
  if (!m_jacobianValues)
  {
    ComputeJacobian();
  }

  if (!m_referenceValues)
  {
    CreateReferenceBuffer();
  }
}

void AlbedoCostFunction::ComputeJacobian()
{
  LYNX_ASSERT(!m_keyframes->Empty(), "no keyframes have been added");

  ResetJacobian();

  const size_t launchSize = m_keyframes->GetValidPixelCount();
  std::shared_ptr<Context> context = m_material->GetContext();
  context->GetVariable("computeAlbedoDerivs")->setUint(true);
  context->Launch(m_programId, launchSize);
  context->GetVariable("computeAlbedoDerivs")->setUint(false);

  CUdeviceptr pointer = m_jacobian->getDevicePointer(0);
  m_jacobianValues = reinterpret_cast<float*>(pointer);
}

void AlbedoCostFunction::ResetJacobian()
{
  const size_t rows = GetResidualCount(); // TODO: check size
  const size_t cols = GetParameterCount(); // TODO: check size
  const size_t count = 3 * rows * cols;
  m_jacobian->setSize(cols, rows);

  float* device = reinterpret_cast<float*>(m_jacobian->map());
  std::fill(device, device + count, 0.0f);
  m_jacobian->unmap();
}

void AlbedoCostFunction::CreateReferenceBuffer()
{
  optix::Buffer reference = m_keyframes->GetReferenceBuffer();
  CUdeviceptr pointer = reference->getDevicePointer(0);
  m_referenceValues = reinterpret_cast<float*>(pointer);

  // TODO: subtrace "render" buffer from reference buffer

  throw std::string("function not implemented: ") + __FUNCTION__;
}

void AlbedoCostFunction::Initialize()
{
  SetDimensions();
  CreateBuffer();
  CreateKeyframeSet();
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
  m_material->SetDerivativeBuffer(m_jacobian);
  m_jacobian->setFormat(RT_FORMAT_FLOAT3);
  m_jacobian->setSize(1, 1);
}

void AlbedoCostFunction::CreateProgram()
{
  std::shared_ptr<Context> context = m_material->GetContext();
  const std::string file = PtxUtil::GetFile("AlbedoCostFunction");
  m_program = context->CreateProgram(file, "Capture");
  m_programId = context->RegisterLaunchProgram(m_program);
}

void AlbedoCostFunction::CreateKeyframeSet()
{
  std::shared_ptr<Context> context = m_material->GetContext();
  m_keyframes = std::make_unique<KeyframeSet>(context);
  m_program["cameras"]->setBuffer(m_keyframes->GetCameraBuffer());
  m_program["pixelSamples"]->setBuffer(m_keyframes->GetPixelBuffer());
  m_program["render"]->setBuffer(m_keyframes->GetRenderBuffer());
}

} // namespace torch