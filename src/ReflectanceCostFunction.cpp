#include <torch/ReflectanceCostFunction.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/device/ReflectanceCostFunction.cuh>

#include <iostream>
#include <eigen3/Eigen/Eigen>

namespace torch
{

ReflectanceCostFunction::ReflectanceCostFunction(
    std::shared_ptr<MatteMaterial> material, std::shared_ptr<Mesh> mesh) :
  m_material(material),
  m_mesh(mesh),
  m_adjacencyMap(nullptr),
  m_adjacencyOffsets(nullptr),
  m_chromThreshold(0.5),
  m_weight(1.0),
  m_rowIndices(nullptr)
{
  Initialize();
}

ReflectanceCostFunction::~ReflectanceCostFunction()
{
  cudaFree(m_adjacencyMap);
  cudaFree(m_adjacencyOffsets);
  cudaFree(m_rowIndices);
}

float ReflectanceCostFunction::GetChromaticityThreshold() const
{
  return m_chromThreshold;
}

void ReflectanceCostFunction::SetChromaticityThreshold(float threshold)
{
  m_chromThreshold = threshold;
}

float ReflectanceCostFunction::GetWeight() const
{
  return m_weight;
}

void ReflectanceCostFunction::SetWeight(float weight)
{
  m_weight = weight;
}

lynx::Matrix* ReflectanceCostFunction::CreateJacobianMatrix()
{
  return new lynx::SparseMatrix3C(m_valueCount, m_rowIndices, m_adjacencyMap,
      GetResidualCount() / 3, GetResidualCount() / 3);
}

void ReflectanceCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  const size_t count = GetResidualCount();

  torch::Evaluate(parameters[0], residuals, nullptr, count, m_adjacencyMap,
      m_adjacencyOffsets, m_chromThreshold, m_weight);
}

void ReflectanceCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  lynx::SparseMatrix3C* matrix;
  matrix = dynamic_cast<lynx::SparseMatrix3C*>(jacobian);
  LYNX_ASSERT(matrix, "expected jacobian to be SparseMatrix3C type");
  LYNX_ASSERT(size == GetResidualCount(), "partial evaluation not supported");

  const float* params = parameters[0];
  float* J = &matrix->GetValues()[offset];
  float* r = &residuals[offset];

  torch::Evaluate(params, r, J, size, m_adjacencyMap, m_adjacencyOffsets,
      m_chromThreshold, m_weight);
}

void ReflectanceCostFunction::Initialize()
{
  SetDimensions();
  CreateAdjacencyMap();
}

void ReflectanceCostFunction::SetDimensions()
{
  const size_t count = 3 * m_material->GetAlbedoCount();
  lynx::CostFunction::m_residualCount = count;
  lynx::CostFunction::m_parameterBlockSizes.push_back(count);
  lynx::CostFunction::m_maxEvaluationBlockSize = count;
}

void ReflectanceCostFunction::CreateAdjacencyMap()
{
  std::vector<uint> map;
  std::vector<uint> offsets;
  m_mesh->GetVertexAdjacencyMap(map, offsets, true);
  m_valueCount = map.size();

  const size_t mapBytes = sizeof(uint) * map.size();
  const size_t offsetsBytes = sizeof(uint) * offsets.size();
  LYNX_CHECK_CUDA(cudaMalloc(&m_adjacencyMap, mapBytes));
  LYNX_CHECK_CUDA(cudaMalloc(&m_adjacencyOffsets, offsetsBytes));

  LYNX_CHECK_CUDA(cudaMemcpy(m_adjacencyMap, map.data(), mapBytes,
      cudaMemcpyHostToDevice));

  LYNX_CHECK_CUDA(cudaMemcpy(m_adjacencyOffsets, offsets.data(), offsetsBytes,
                             cudaMemcpyHostToDevice));


  std::vector<uint> rowIndices(map.size());
  uint index = 0;

  for (size_t i = 0; i < offsets.size() - 1; ++i)
  {
    const uint count = offsets[i + 1] - offsets[i];

    for (uint j = 0; j < count; ++j)
    {
      rowIndices[index++] = i;
    }
  }

  LYNX_CHECK_CUDA(cudaMalloc(&m_rowIndices, mapBytes));

  LYNX_CHECK_CUDA(cudaMemcpy(m_rowIndices, rowIndices.data(), mapBytes,
      cudaMemcpyHostToDevice));
}

} // namespace torch