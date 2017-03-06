#include <torch/MeshCostFunction.h>
#include <torch/Context.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/PtxUtil.h>
#include <torch/VoxelLight.h>

namespace torch
{

MeshCostFunction::MeshCostFunction(std::shared_ptr<VoxelLight> light,
    std::shared_ptr<Mesh> mesh, std::shared_ptr<MatteMaterial> material) :
  m_locked(false),
  m_light(light),
  m_mesh(mesh),
  m_material(material),
  m_lightCoeffValues(nullptr),
  m_referenceValues(nullptr),
  m_adjacentVertices(nullptr),
  m_adjacentWeights(nullptr),
  m_iterations(0),
  m_maxNeighborDistance(0.1f),
  m_similarityThreshold(0.1f)
{
  Initialize();
}

MeshCostFunction::~MeshCostFunction()
{
  cudaFree(m_referenceValues);
  cudaFree(m_adjacentVertices);
  cudaFree(m_adjacentWeights);
}

void MeshCostFunction::SetMaxNeighborDistance(float distance)
{
  LYNX_ASSERT(!m_locked, "adjacencies cannot be updated");
  m_maxNeighborDistance = distance;
  ClearAdjacencies();
}

void MeshCostFunction::SetSimilarityThreshold(float threshold)
{
  LYNX_ASSERT(!m_locked, "adjacencies cannot be updated");
  m_similarityThreshold = threshold;
  ClearAdjacencies();
}

lynx::Matrix* MeshCostFunction::CreateJacobianMatrix()
{
  m_locked = true;
  PrepareEvaluation();
  const size_t rows = m_adjacencyCount;
  const size_t cols = m_light->GetVoxelCount();
  return new lynx::Matrix3C(m_lightCoeffValues, rows, cols);
}

void MeshCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  // TODO: take care of divide by zero error!!!

  PrepareEvaluation();
  // const size_t rows = m_adjacencyCount;
  // const size_t cols = m_light->GetVoxelCount();
  // lynx::Matrix3C jacobian(m_lightCoeffValues, rows, cols);
  // jacobian.RightMultiply(parameters[0], residuals);
  // lynx::Scale(-1.0f, residuals, residuals, GetResidualCount());
  // lynx::Log(residuals, residuals, GetResidualCount());
  // lynx::Sub(m_referenceValues, residuals, residuals, GetResidualCount());
}

void MeshCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  // TODO: take care of divide by zero error!!!

  // LYNX_ASSERT(jacobian->GetValues() == m_lightCoeffValues, "invalid jacobian");
  // LYNX_ASSERT(offset == 0, "sub-problem cannot be evaluated");
  // LYNX_ASSERT(size == GetResidualCount(), "sub-problem cannot be evaluated");

  // PrepareEvaluation();
  // jacobian->RightMultiply(parameters[0], residuals);
  // lynx::Scale(-1.0f, residuals, residuals, GetResidualCount());
  // lynx::Log(residuals, residuals, GetResidualCount());
  // lynx::Sub(m_referenceValues, residuals, residuals, GetResidualCount());
}

void MeshCostFunction::ClearJacobian()
{
  m_lightCoeffValues = nullptr;
}

void MeshCostFunction::PrepareEvaluation()
{
  if (!m_adjacentVertices)
  {
    ComputeAdjacenies();
  }

  if (!m_lightCoeffValues)
  {
    ComputeLightCoefficients();
  }
}

void MeshCostFunction::ComputeAdjacenies()
{
  // TODO: implement

  // m_adjacencyCount
  // m_adjacencyVertices
  // m_adjacencyWeights

  lynx::CostFunction::m_maxEvaluationBlockSize = 3 * m_adjacencyCount;
  lynx::CostFunction::m_residualCount = 3 * m_adjacencyCount;
}

void MeshCostFunction::ComputeLightCoefficients()
{
  ResetLightCoefficients();

  const size_t launchSize = m_mesh->GetVertexCount();
  std::shared_ptr<Context> context = m_light->GetContext();
  m_program["iteration"]->setUint(m_iterations++);
  context->GetVariable("computeVoxelDerivs")->setUint(true);
  context->Launch(m_programId, launchSize);
  context->GetVariable("computeVoxelDerivs")->setUint(false);

  CUdeviceptr pointer = m_lightCoeffs->getDevicePointer(0);
  m_lightCoeffValues = reinterpret_cast<float*>(pointer);
}

void MeshCostFunction::ResetLightCoefficients()
{
  const size_t rows = m_adjacencyCount;
  const size_t cols = m_light->GetVoxelCount();
  const size_t count = 3 * rows * cols;

  float* device = reinterpret_cast<float*>(m_lightCoeffs->map());
  std::fill(device, device + count, 0.0f);
  m_lightCoeffs->unmap();
}

void MeshCostFunction::ClearAdjacencies()
{
  LYNX_CHECK_CUDA(cudaFree(m_adjacentVertices));
  LYNX_CHECK_CUDA(cudaFree(m_adjacentWeights));
  m_adjacentVertices = nullptr;
  m_adjacentWeights = nullptr;
}

void MeshCostFunction::Initialize()
{
  SetDimensions();
  CreateBuffer();
  CreateProgram();
  ComputeReferenceValues();
}

void MeshCostFunction::SetDimensions()
{
  const size_t paramCount = 3 * m_light->GetVoxelCount();
  lynx::CostFunction::m_parameterBlockSizes.push_back(paramCount);
  lynx::CostFunction::m_maxEvaluationBlockSize = 0;
  lynx::CostFunction::m_residualCount = 0;
}

void MeshCostFunction::CreateBuffer()
{
  const size_t residCount = m_mesh->GetVertexCount();
  const size_t paramCount = m_light->GetVoxelCount();
  std::shared_ptr<Context> context = m_light->GetContext();
  m_lightCoeffs = context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_light->SetDerivativeBuffer(m_lightCoeffs);
  m_lightCoeffs->setFormat(RT_FORMAT_FLOAT3);
  m_lightCoeffs->setSize(residCount, paramCount);
}

void MeshCostFunction::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("MeshCostFunction");
  std::shared_ptr<Context> context = m_light->GetContext();
  m_program = context->CreateProgram(file, "Capture");
  m_programId = context->RegisterLaunchProgram(m_program);
}

void MeshCostFunction::ComputeReferenceValues()
{
  optix::Buffer albedos = m_material->GetAlbedoBuffer();
  CUdeviceptr pointer = albedos->getDevicePointer(0);
  float* referenceValues = reinterpret_cast<float*>(pointer);

  const size_t count = GetResidualCount();
  const size_t bytes = sizeof(float) * count;
  LYNX_CHECK_CUDA(cudaMalloc(&m_referenceValues, bytes));
  lynx::Log(referenceValues, m_referenceValues, count);
}

} // namespace torch