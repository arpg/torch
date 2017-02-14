#include <gtest/gtest.h>
#include <torch/Torch.h>
#include <lynx/lynx.h>
#include <iostream>

namespace torch
{

namespace testing
{

TEST(ActivationCostFunction, General)
{
  Scene scene;

  std::shared_ptr<EnvironmentLight> light;
  light = scene.CreateEnvironmentLight();
  light->SetRowCount(3);
  light->SetRadiance(0.1f, 0.1f, 0.1f);
  scene.Add(light);

  Image image;
  std::shared_ptr<Camera> camera;
  camera = scene.CreateCamera();
  camera->Capture(image);

  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);

  float* values = reinterpret_cast<float*>(pointer);
  std::vector<float> radiance(3 * light->GetDirectionCount());

  lynx::Problem problem;

  problem.AddParameterBlock(values, radiance.size());

  torch::ActivationCostFunction* costFunction;
  costFunction = new torch::ActivationCostFunction(light);
  problem.AddResidualBlock(costFunction, nullptr, values);

  problem.CheckGradients();
}

} // namespace testing

} // namespace torch