#include <torch/LightCostFunction.h>
#include <torch/Context.h>
#include <torch/EnvironmentLight.h>
#include <torch/Exception.h>
#include <torch/Keyframe.h>
#include <torch/KeyframeSet.h>
#include <torch/PtxUtil.h>
#include <torch/Spectrum.h>

namespace torch
{

LightCostFunction::LightCostFunction(
    std::shared_ptr<EnvironmentLight> light) :
  m_locked(false),
  m_light(light),
  m_jacobianValues(nullptr),
  m_referenceValues(nullptr),
  m_iterations(0)
{
  Initialize();
}

LightCostFunction::~LightCostFunction()
{
}

void LightCostFunction::AddKeyframe(std::shared_ptr<Keyframe> keyframe)
{
  LYNX_ASSERT(!m_locked, "new keyframes cannot be added");
  const size_t residualCount = 3 * keyframe->GetValidPixelCount();
  lynx::CostFunction::m_residualCount += residualCount;
  lynx::CostFunction::m_maxEvaluationBlockSize += residualCount;
  m_referenceValues = nullptr;
  m_jacobianValues = nullptr;
  m_keyframes->Add(keyframe);
}

lynx::Matrix* LightCostFunction::CreateJacobianMatrix()
{
  m_locked = true;
  PrepareEvaluation();
  const size_t rows = GetResidualCount() / 3;
  const size_t cols = GetParameterCount() / 3;
  return new lynx::Matrix3C(m_jacobianValues, rows, cols);
}

void LightCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  PrepareEvaluation();
  const size_t rows = GetResidualCount() / 3;
  const size_t cols = GetParameterCount() / 3;
  lynx::Matrix3C jacobian(m_jacobianValues, rows, cols);
  jacobian.RightMultiply(parameters[0], residuals);
  lynx::Add(m_referenceValues, residuals, residuals, GetResidualCount());
}

void LightCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  LYNX_ASSERT(jacobian->GetValues() == m_jacobianValues, "invalid jacobian");
  LYNX_ASSERT(offset == 0, "sub-problem cannot be evaluated");
  LYNX_ASSERT(size == GetResidualCount(), "sub-problem cannot be evaluated");

  PrepareEvaluation();
  jacobian->RightMultiply(parameters[0], residuals);
  lynx::Add(m_referenceValues, residuals, residuals, GetResidualCount());
}

void LightCostFunction::Evaluate(const float* const* parameters,
    float* residuals, float* gradient)
{
  PrepareEvaluation();
  const size_t rows = GetResidualCount() / 3;
  const size_t cols = GetParameterCount() / 3;
  lynx::Matrix3C jacobian(m_jacobianValues, rows, cols);
  jacobian.RightMultiply(parameters[0], residuals);
  lynx::Add(m_referenceValues, residuals, residuals, GetResidualCount());
  jacobian.LeftMultiply(residuals, gradient);
}

void LightCostFunction::ClearJacobian()
{
  m_jacobianValues = nullptr;
}

void LightCostFunction::PrepareEvaluation()
{
  if (!m_referenceValues)
  {
    CreateReferenceBuffer();
  }

  if (!m_jacobianValues)
  {
    ComputeJacobian();
  }
}

void LightCostFunction::CreateReferenceBuffer()
{
  optix::Buffer reference = m_keyframes->GetReferenceBuffer();
  CUdeviceptr pointer = reference->getDevicePointer(0);
  m_referenceValues = reinterpret_cast<float*>(pointer);
}

void LightCostFunction::ComputeJacobian()
{
  LYNX_ASSERT(!m_keyframes->Empty(), "no keyframes have been added");

  ResetJacobian();

  const size_t launchSize = m_keyframes->GetValidPixelCount();
  std::shared_ptr<Context> context = m_light->GetContext();
  m_program["iteration"]->setUint(m_iterations++);
  context->GetVariable("computeLightDerivs")->setUint(true);
  context->Launch(m_programId, launchSize);
  context->GetVariable("computeLightDerivs")->setUint(false);

  CUdeviceptr pointer = m_jacobian->getDevicePointer(0);
  m_jacobianValues = reinterpret_cast<float*>(pointer);
}

void LightCostFunction::ResetJacobian()
{
  const size_t rows = GetResidualCount() / 3;
  const size_t cols = GetParameterCount() / 3;
  const size_t count = 3 * rows * cols;
  m_jacobian->setSize(rows, cols);

  float* device = reinterpret_cast<float*>(m_jacobian->map());
  std::fill(device, device + count, 0.0f);
  m_jacobian->unmap();
}

void LightCostFunction::Initialize()
{
  SetDimensions();
  CreateBuffer();
  CreateProgram();
  CreateKeyframeSet();
}

void LightCostFunction::SetDimensions()
{
  const size_t paramCount = 3 * m_light->GetDirectionCount();
  lynx::CostFunction::m_parameterBlockSizes.push_back(paramCount);
  lynx::CostFunction::m_maxEvaluationBlockSize = 0;
  lynx::CostFunction::m_residualCount = 0;
}

void LightCostFunction::CreateBuffer()
{
  std::shared_ptr<Context> context = m_light->GetContext();
  m_jacobian = context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_light->SetDerivativeBuffer(m_jacobian);
  m_jacobian->setFormat(RT_FORMAT_FLOAT3);
  m_jacobian->setSize(1, 1);
}

void LightCostFunction::CreateProgram()
{
  std::shared_ptr<Context> context = m_light->GetContext();
  const std::string file = PtxUtil::GetFile("LightCostFunction");
  m_program = context->CreateProgram(file, "Capture");
  m_programId = context->RegisterLaunchProgram(m_program);
}

void LightCostFunction::CreateKeyframeSet()
{
  std::shared_ptr<Context> context = m_light->GetContext();
  m_keyframes = std::make_unique<KeyframeSet>(context);
  m_program["cameras"]->setBuffer(m_keyframes->GetCameraBuffer());
  m_program["pixelSamples"]->setBuffer(m_keyframes->GetPixelBuffer());
  m_program["render"]->setBuffer(m_keyframes->GetRenderBuffer());
}

} // namespace torch