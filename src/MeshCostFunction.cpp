#include <torch/MeshCostFunction.h>
#include <torch/Context.h>
#include <torch/Exception.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/Normal.h>
#include <torch/Octree.h>
#include <torch/Point.h>
#include <torch/PtxUtil.h>
#include <torch/Spectrum.h>
#include <torch/VoxelLight.h>
#include <torch/device/MeshCostKernel.cuh>
#include <torch/device/Ray.h>

#include <iostream>
#include <torch/MeshWriter.h>

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
  m_shadingValues(nullptr),
  m_plist(nullptr),
  m_qlist(nullptr),
  m_adjacentWeights(nullptr),
  m_iterations(0),
  m_sampleCount(1),
  m_maxNeighborCount(1),
  m_maxNeighborDistance(0.1f),
  m_similarityThreshold(0.1f)
{
  Initialize();
}

MeshCostFunction::~MeshCostFunction()
{
  cudaFree(m_referenceValues);
  cudaFree(m_shadingValues);
  cudaFree(m_plist);
  cudaFree(m_qlist);
  cudaFree(m_adjacentWeights);
}

void MeshCostFunction::SetLightSampleCount(unsigned int count)
{
  m_sampleCount = count;
  m_program["sampleCount"]->setUint(m_sampleCount);
  m_closestHitProgram["sampleCount"]->setUint(m_sampleCount);
}

void MeshCostFunction::SetMaxNeighborCount(unsigned int count)
{
  LYNX_ASSERT(!m_locked, "adjacencies cannot be updated");
  LYNX_ASSERT(count > 0, "count cannot be zero");
  m_maxNeighborCount = count;
  ClearAdjacencies();
  ClearJacobians();
}

void MeshCostFunction::SetMaxNeighborDistance(float distance)
{
  LYNX_ASSERT(!m_locked, "adjacencies cannot be updated");
  m_maxNeighborDistance = distance;
  ClearAdjacencies();
  ClearJacobians();
}

void MeshCostFunction::SetSimilarityThreshold(float threshold)
{
  LYNX_ASSERT(!m_locked, "adjacencies cannot be updated");
  m_similarityThreshold = threshold;
  ClearAdjacencies();
  ClearJacobians();
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
  PrepareEvaluation();

  const size_t rows = m_mesh->GetVertexCount();
  const size_t cols = m_light->GetVoxelCount();
  lynx::SetZeros(m_shadingValues, 3 * rows);
  lynx::Matrix3C coeffs(m_lightCoeffValues, rows, cols);
  coeffs.RightMultiply(parameters[0], m_shadingValues);
  lynx::Scale(-1.0, m_shadingValues, m_shadingValues, 3 * rows);

  torch::EvaluateR(m_plist, m_qlist, m_adjacentWeights, m_referenceValues,
           m_shadingValues, residuals, 3 * m_adjacencyCount);
}

void MeshCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  TORCH_THROW("not implemented");
}

void MeshCostFunction::Evaluate(const float* const* parameters,
    float* residuals, float* gradient)
{
  PrepareEvaluation();

  const size_t rows = m_mesh->GetVertexCount();
  const size_t cols = m_light->GetVoxelCount();
  lynx::SetZeros(m_shadingValues, 3 * rows);
  lynx::Matrix3C coeffs(m_lightCoeffValues, rows, cols);
  coeffs.RightMultiply(parameters[0], m_shadingValues);
  lynx::Scale(-1.0, m_shadingValues, m_shadingValues, 3 * rows);

  // std::vector<Spectrum> albedos(m_material->GetAlbedoCount());
  // std::shared_ptr<MatteMaterial> material;
  // material = std::make_shared<MatteMaterial>(m_material->GetContext());

  // const size_t bytes = sizeof(Spectrum) * albedos.size();
  // const cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
  // LYNX_CHECK_CUDA(cudaMemcpy(albedos.data(), m_shadingValues, bytes, kind));
  // material->SetAlbedos(albedos);

  // MeshWriter writer(m_mesh, material);
  // writer.SetMeshlabFriendly(true);
  // writer.Write("shading_values.ply");

  torch::EvaluateR(m_plist, m_qlist, m_adjacentWeights, m_referenceValues,
           m_shadingValues, residuals, 3 * m_adjacencyCount);

  // std::cout << std::endl;
  // std::cout << "Light Coefficients:" << std::endl;
  // std::cout << std::endl;

  // for (size_t y = 0; y < rows; ++y)
  // {
  //   for (size_t x = 0; x < cols; ++x)
  //   {
  //     const size_t index = x * rows + y;

  //     std::cout << "( ";
  //     std::cout << lynx::Get(m_lightCoeffValues + (3 * index + 0)) << " ";
  //     std::cout << lynx::Get(m_lightCoeffValues + (3 * index + 1)) << " ";
  //     std::cout << lynx::Get(m_lightCoeffValues + (3 * index + 2)) << " ";
  //     std::cout << ") ";
  //   }

  //   std::cout << std::endl;
  // }

  // std::cout << std::endl;
  // std::cout << "Parameters:" << std::endl;
  // std::cout << std::endl;

  // for (size_t i = 0; i < m_light->GetVoxelCount(); ++i)
  // {
  //   std::cout << "( ";
  //   std::cout << lynx::Get(parameters[0] + (3 * i + 0)) << " ";
  //   std::cout << lynx::Get(parameters[0] + (3 * i + 1)) << " ";
  //   std::cout << lynx::Get(parameters[0] + (3 * i + 2)) << " ";
  //   std::cout << ")" << std::endl;
  // }

  // std::cout << std::endl;
  // std::cout << "Shading Values:" << std::endl;
  // std::cout << std::endl;

  // for (size_t i = 0; i < m_material->GetAlbedoCount(); ++i)
  // {
  //   std::cout << "( ";
  //   std::cout << lynx::Get(m_shadingValues + (3 * i + 0)) << " ";
  //   std::cout << lynx::Get(m_shadingValues + (3 * i + 1)) << " ";
  //   std::cout << lynx::Get(m_shadingValues + (3 * i + 2)) << " ";
  //   std::cout << ")" << std::endl;
  // }

  // std::cout << std::endl;
  // std::cout << "Residuals:" << std::endl;
  // std::cout << std::endl;

  // for (size_t i = 0; i < m_adjacencyCount; ++i)
  // {
  //   std::cout << "( ";
  //   std::cout << lynx::Get(residuals + (3 * i + 0)) << " ";
  //   std::cout << lynx::Get(residuals + (3 * i + 1)) << " ";
  //   std::cout << lynx::Get(residuals + (3 * i + 2)) << " ";
  //   std::cout << ")" << std::endl;
  // }

  // std::cout << std::endl;
  // std::cout << "Jacobian:" << std::endl;
  // std::cout << std::endl;

  size_t offset = 0;
  const size_t maxEval = 1000;
  const size_t residualCount = m_adjacencyCount;
  const size_t stepCount = (residualCount - 1) / maxEval;

  float* jacobian = m_maxJacobian->GetValues();

  for (size_t step = 0; step < stepCount; ++step)
  {
    torch::EvaluateJ(&m_plist[offset], &m_qlist[offset],
        m_adjacentWeights, m_referenceValues, m_shadingValues,
        m_lightCoeffValues, jacobian, m_adjacencyCount,
        m_light->GetVoxelCount(), m_mesh->GetVertexCount(), maxEval);

    m_maxJacobian->LeftMultiply(residuals + offset, gradient);
    offset += maxEval;
  }

  jacobian = m_minJacobian->GetValues();
  const size_t minEval = residualCount - maxEval * (residualCount / maxEval);

  torch::EvaluateJ(&m_plist[offset], &m_qlist[offset],
      m_adjacentWeights, m_referenceValues, m_shadingValues,
      m_lightCoeffValues, jacobian, m_adjacencyCount,
      m_light->GetVoxelCount(), m_mesh->GetVertexCount(), minEval);

  m_minJacobian->LeftMultiply(residuals + offset, gradient);

  // for (size_t y = 0; y < m_adjacencyCount; ++y)
  // {
  //   for (size_t x = 0; x < m_light->GetVoxelCount(); ++x)
  //   {
  //     const size_t index = x * m_adjacencyCount + y;
  //     float r = lynx::Get(jacobian + (3 * index + 0));
  //     float g = lynx::Get(jacobian + (3 * index + 1));
  //     float b = lynx::Get(jacobian + (3 * index + 2));

  //     if (r > -1E-10 && r < +1E-10) r = +0.0f;
  //     if (g > -1E-10 && g < +1E-10) g = +0.0f;
  //     if (b > -1E-10 && b < +1E-10) b = +0.0f;

  //     std::cout << "( ";
  //     printf("%s%0.4f ", ((r < 0) ? "" : " "), r);
  //     printf("%s%0.4f ", ((g < 0) ? "" : " "), g);
  //     printf("%s%0.4f ", ((b < 0) ? "" : " "), b);
  //     std::cout << ") ";
  //   }

  //   std::cout << std::endl;
  // }

  // std::cout << std::endl;

  ClearJacobian();
}

void MeshCostFunction::ClearJacobian()
{
  m_lightCoeffValues = nullptr;
}

void MeshCostFunction::PrepareEvaluation()
{
  if (!m_plist)
  {
    ComputeAdjacenies();
  }

  if (!m_lightCoeffValues)
  {
    ComputeLightCoefficients();
  }

  if (!m_maxJacobian)
  {
    AllocateJacobians();
  }
}

void MeshCostFunction::ComputeAdjacenies()
{
  Octree octree(10.0f, 10);
  std::vector<unsigned int> plist;
  std::vector<unsigned int> qlist;
  std::vector<float> weights;

  std::vector<Point> vertices;
  m_mesh->GetVertices(vertices);

  std::vector<Normal> normals;
  m_mesh->GetNormals(normals);

  std::vector<Spectrum> albedos;
  m_material->GetAlbedos(albedos);

  std::cout << "Populating octree..." << std::endl;

  for (size_t i = 0; i < vertices.size(); ++i)
  {
    const Point& v = vertices[i];
    const float3 position = make_float3(v.x, v.y, v.z);
    octree.AddVertex(i, position);
  }

  std::cout << "Finding adjacencies..." << std::endl;

  for (size_t p = 0; p < vertices.size(); ++p)
  {
    if (vertices.size() < 10 || p % (vertices.size() / 10) == 0)
    {
      std::cout << (100.0 * p) / vertices.size() << "%" << std::endl;
    }

    std::vector<unsigned int> neighbors;

    const Point& vp = vertices[p];
    const Normal& np = normals[p];
    const Spectrum& ap = albedos[p];
    const Vector cp = ap.GetRGB().Normalize();

    const float3 position = make_float3(vp.x, vp.y, vp.z);
    octree.GetVertices(p + 1, position, m_maxNeighborDistance, neighbors);

    Spectrum totalColor;
    float totalWeight = 0;
    std::vector<unsigned int> ppot;
    std::vector<unsigned int> qpot;
    std::vector<float> wpot;
    unsigned int added = 0;

    for (size_t j = 0; j < neighbors.size(); ++j)
    {
      const size_t q = neighbors[j];
      const Point& vq = vertices[q];
      const Normal& nq = normals[q];
      const Spectrum& aq = albedos[q];
      const Vector cq = aq.GetRGB().Normalize();

      float similarity = 1.0f;
      similarity *= 1 - ((vp - vq).Length() / m_maxNeighborDistance);
      similarity *= np * nq;
      similarity *= cp * cq;

      if (similarity > m_similarityThreshold)
      {
        // plist.push_back(p);
        // qlist.push_back(q);
        // weights.push_back(similarity);

        ppot.push_back(p);
        qpot.push_back(q);
        wpot.push_back(similarity);
        totalColor += albedos[q];
        totalWeight += similarity;

        if (++added == m_maxNeighborCount) break;
      }
    }

    const float meanWeight = totalWeight / added;
    const Spectrum meanColor = totalColor / added;
    const float colorDiff = meanColor.GetY() - albedos[p].GetY();

    if (colorDiff > 0)
    {
      for (unsigned int i = 0; i < added; ++i)
      {
        plist.push_back(ppot[i]);
        qlist.push_back(qpot[i]);
        // weights.push_back(colorDiff * meanWeight * wpot[i]);
        weights.push_back(wpot[i]);
      }
    }
  }

  m_adjacencyCount = plist.size();
  TORCH_ASSERT(m_adjacencyCount > 0, "not vertex pairs found");

  std::cout << "Found " << m_adjacencyCount << " adjacencies" << std::endl;

  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  size_t bytes = sizeof(unsigned int) * m_adjacencyCount;
  LYNX_CHECK_CUDA(cudaMalloc(&m_plist, bytes));
  LYNX_CHECK_CUDA(cudaMalloc(&m_qlist, bytes));
  LYNX_CHECK_CUDA(cudaMemcpy(m_plist, plist.data(), bytes, kind));
  LYNX_CHECK_CUDA(cudaMemcpy(m_qlist, qlist.data(), bytes, kind));

  bytes = sizeof(float) * m_adjacencyCount;
  LYNX_CHECK_CUDA(cudaMalloc(&m_adjacentWeights, bytes));
  LYNX_CHECK_CUDA(cudaMemcpy(m_adjacentWeights, weights.data(), bytes, kind));

  // TODO: actually check memory usage
  lynx::CostFunction::m_maxEvaluationBlockSize = 1000;
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
  const size_t rows = m_mesh->GetVertexCount();
  const size_t cols = m_light->GetVoxelCount();
  const size_t count = 3 * rows * cols;

  float* device = reinterpret_cast<float*>(m_lightCoeffs->map());
  std::fill(device, device + count, 0.0f);
  m_lightCoeffs->unmap();
}

void MeshCostFunction::AllocateJacobians()
{
  const size_t residualCount = m_adjacencyCount;
  const size_t paramCount = m_light->GetVoxelCount();
  const size_t maxEval = 1000; // TODO: actually check memory usage
  const size_t minEval = residualCount - maxEval * (residualCount / maxEval);
  m_maxJacobian = std::make_unique<lynx::Matrix3C>(maxEval, paramCount);
  m_minJacobian = std::make_unique<lynx::Matrix3C>(minEval, paramCount);
}

void MeshCostFunction::ClearAdjacencies()
{
  LYNX_CHECK_CUDA(cudaFree(m_plist));
  LYNX_CHECK_CUDA(cudaFree(m_qlist));
  LYNX_CHECK_CUDA(cudaFree(m_adjacentWeights));
  m_plist = nullptr;
  m_qlist = nullptr;
  m_adjacentWeights = nullptr;
}

void MeshCostFunction::ClearJacobians()
{
  m_maxJacobian = nullptr;
  m_minJacobian = nullptr;
}

void MeshCostFunction::Initialize()
{
  SetDimensions();
  CreateDummyMaterial();
  CreateDummyGeometry();
  CreateShadingBuffer();
  CreateShadingProgram();
  ComputeReferenceValues();
  AllocateSharingValues();
}

void MeshCostFunction::SetDimensions()
{
  const size_t paramCount = 3 * m_light->GetVoxelCount();
  lynx::CostFunction::m_parameterBlockSizes.push_back(paramCount);
  lynx::CostFunction::m_maxEvaluationBlockSize = 0;
  lynx::CostFunction::m_residualCount = 0;
}

void MeshCostFunction::CreateDummyMaterial()
{
  std::shared_ptr<Context> context = m_mesh->GetContext();
  const std::string file = PtxUtil::GetFile("MeshCostFunction");
  m_closestHitProgram = context->CreateProgram(file, "ClosestHit");
  m_closestHitProgram["normals"]->setBuffer(m_mesh->GetNormalBuffer());
  m_closestHitProgram["vertices"]->setBuffer(m_mesh->GetVertexBuffer());
  m_closestHitProgram["sampleCount"]->setUint(m_sampleCount);
  m_dummyMaterial = context->CreateMaterial();
  m_dummyMaterial->setClosestHitProgram(RAY_TYPE_RADIANCE, m_closestHitProgram);
}

void MeshCostFunction::CreateDummyGeometry()
{
  std::shared_ptr<Context> context = m_mesh->GetContext();
  const std::string file = PtxUtil::GetFile("MeshCostFunction");
  optix::Program boundsProgram = context->CreateProgram(file, "GetBounds");
  optix::Program intersectProgram = context->CreateProgram(file, "Intersect");

  boundsProgram["normals"]->setBuffer(m_mesh->GetNormalBuffer());
  boundsProgram["vertices"]->setBuffer(m_mesh->GetVertexBuffer());
  intersectProgram["normals"]->setBuffer(m_mesh->GetNormalBuffer());
  intersectProgram["vertices"]->setBuffer(m_mesh->GetVertexBuffer());

  m_dummyGeometry = context->CreateGeometry();
  m_dummyGeometry->setBoundingBoxProgram(boundsProgram);
  m_dummyGeometry->setIntersectionProgram(intersectProgram);
  m_dummyGeometry->setPrimitiveCount(1);

  m_dummyInstance = context->CreateGeometryInstance();
  m_dummyInstance->setGeometry(m_dummyGeometry);
  m_dummyInstance->addMaterial(m_dummyMaterial);

  m_dummyAccel = context->CreateAcceleration();
  m_dummyAccel->setBuilder("NoAccel");
  m_dummyAccel->setTraverser("NoAccel");

  m_dummyGroup = context->CreateGeometryGroup();
  m_dummyGroup->setAcceleration(m_dummyAccel);
  m_dummyGroup->addChild(m_dummyInstance);
}

void MeshCostFunction::CreateShadingBuffer()
{
  const size_t residCount = m_mesh->GetVertexCount();
  const size_t paramCount = m_light->GetVoxelCount();
  std::shared_ptr<Context> context = m_light->GetContext();
  m_lightCoeffs = context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_light->SetDerivativeBuffer(m_lightCoeffs);
  m_lightCoeffs->setFormat(RT_FORMAT_FLOAT3);
  m_lightCoeffs->setSize(residCount, paramCount);
}

void MeshCostFunction::CreateShadingProgram()
{
  const std::string file = PtxUtil::GetFile("MeshCostFunction");
  std::shared_ptr<Context> context = m_light->GetContext();
  m_program = context->CreateProgram(file, "Capture");
  m_programId = context->RegisterLaunchProgram(m_program);
  m_program["vertices"]->setBuffer(m_mesh->GetVertexBuffer());
  m_program["normals"]->setBuffer(m_mesh->GetNormalBuffer());
  m_program["sampleCount"]->setUint(m_sampleCount);
  m_program["dummyRoot"]->set(m_dummyGroup);
}

void MeshCostFunction::ComputeReferenceValues()
{
  optix::Buffer albedos = m_material->GetAlbedoBuffer();
  CUdeviceptr pointer = albedos->getDevicePointer(0);
  float* referenceValues = reinterpret_cast<float*>(pointer);
  lynx::Clamp(referenceValues, 1E-6, 1.0f, 3 * m_material->GetAlbedoCount());

  const size_t count = 3 * m_material->GetAlbedoCount();
  const size_t bytes = sizeof(float) * count;
  LYNX_CHECK_CUDA(cudaMalloc(&m_referenceValues, bytes));
  lynx::Log(referenceValues, m_referenceValues, count);
}

void MeshCostFunction::AllocateSharingValues()
{
  const size_t count = 3 * m_mesh->GetVertexCount();
  const size_t bytes = sizeof(float) * count;
  LYNX_CHECK_CUDA(cudaMalloc(&m_shadingValues, bytes));
}

} // namespace torch