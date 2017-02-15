#include <gtest/gtest.h>
#include <torch/Torch.h>
#include <lynx/lynx.h>

namespace torch
{

namespace testing
{

TEST(ActivationCostFunction, Gradient)
{
  Scene scene;

  std::shared_ptr<EnvironmentLight> light;
  light = scene.CreateEnvironmentLight();
  light->SetRowCount(3);
  light->SetRadiance(0.1f, 0.1f, 0.1f);
  scene.Add(light);

  light->GetContext()->Compile();

  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);

  float* values = reinterpret_cast<float*>(pointer);
  std::vector<float> radiance(3 * light->GetDirectionCount());

  lynx::Problem problem;

  problem.AddParameterBlock(values, radiance.size());
  problem.SetLowerBound(values, 0.0f);

  torch::ActivationCostFunction* costFunction;
  costFunction = new torch::ActivationCostFunction(light);
  problem.AddResidualBlock(costFunction, nullptr, values);

  problem.CheckGradients();
}

TEST(ActivationCostFunction, Optimization)
{
  Scene scene;

  std::shared_ptr<EnvironmentLight> light;
  light = scene.CreateEnvironmentLight();
  light->SetRowCount(3);
  light->SetRadiance(10.1f, 10.1f, 10.1f);
  scene.Add(light);

  light->GetContext()->Compile();

  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);

  float* values = reinterpret_cast<float*>(pointer);
  std::vector<float> radiance(3 * light->GetDirectionCount());

  lynx::Problem problem;

  problem.AddParameterBlock(values, radiance.size());
  problem.SetLowerBound(values, 0.0f);

  torch::ActivationCostFunction* costFunction;
  costFunction = new torch::ActivationCostFunction(light);
  problem.AddResidualBlock(costFunction, nullptr, values);

  lynx::Solver::Summary summary;
  lynx::Solver solver(&problem);
  solver.Solve(&summary);

  ASSERT_TRUE(summary.solutionUsable && summary.finalCost < 1E-6);
}

} // namespace testing

} // namespace torch