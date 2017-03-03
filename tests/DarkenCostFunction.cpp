#include <gtest/gtest.h>
#include <torch/Torch.h>
#include <lynx/lynx.h>

namespace torch
{

namespace testing
{

TEST(DarkenCostFunction, Gradient)
{
  std::vector<float> hValues;
  hValues.push_back(10);
  hValues.push_back(2);
  hValues.push_back(5);
  hValues.push_back(8);
  hValues.push_back(12);

  std::vector<float> hNewValues;
  hNewValues.push_back(9);
  hNewValues.push_back(2);
  hNewValues.push_back(4);
  hNewValues.push_back(9);
  hNewValues.push_back(9);

  const size_t bytes = sizeof(float) * hValues.size();
  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;

  float* values;
  LYNX_CHECK_CUDA(cudaMalloc(&values, bytes));
  LYNX_CHECK_CUDA(cudaMemcpy(values, hValues.data(), bytes, kind))

  float* newValues;
  LYNX_CHECK_CUDA(cudaMalloc(&newValues, bytes));
  LYNX_CHECK_CUDA(cudaMemcpy(newValues, hNewValues.data(), bytes, kind))

  lynx::Problem problem;

  problem.AddParameterBlock(newValues, hNewValues.size());

  torch::DarkenCostFunction* costFunction;
  costFunction = new torch::DarkenCostFunction(hValues.size());
  costFunction->SetValues(values);
  problem.AddResidualBlock(costFunction, nullptr, newValues);

  problem.CheckGradients();
  LYNX_CHECK_CUDA(cudaFree(values));
  LYNX_CHECK_CUDA(cudaFree(newValues));
}

TEST(DarkenCostFunction, Optimization)
{
  std::vector<float> hValues;
  hValues.push_back(10);
  hValues.push_back(2);
  hValues.push_back(5);
  hValues.push_back(8);
  hValues.push_back(12);

  std::vector<float> hNewValues;
  hNewValues.push_back(9);
  hNewValues.push_back(2);
  hNewValues.push_back(4);
  hNewValues.push_back(9);
  hNewValues.push_back(9);

  const size_t bytes = sizeof(float) * hValues.size();
  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;

  float* values;
  LYNX_CHECK_CUDA(cudaMalloc(&values, bytes));
  LYNX_CHECK_CUDA(cudaMemcpy(values, hValues.data(), bytes, kind))

  float* newValues;
  LYNX_CHECK_CUDA(cudaMalloc(&newValues, bytes));
  LYNX_CHECK_CUDA(cudaMemcpy(newValues, hNewValues.data(), bytes, kind))

  lynx::Problem problem;

  problem.AddParameterBlock(newValues, hNewValues.size());

  torch::DarkenCostFunction* costFunction;
  costFunction = new torch::DarkenCostFunction(hValues.size());
  costFunction->SetValues(values);
  problem.AddResidualBlock(costFunction, nullptr, newValues);

  lynx::Solver::Summary summary;
  lynx::Solver solver(&problem);
  solver.Solve(&summary);

  LYNX_CHECK_CUDA(cudaFree(values));
  LYNX_CHECK_CUDA(cudaFree(newValues));
}

} // namespace testing

} // namespace torch