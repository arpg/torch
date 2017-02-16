#include <cmath>
#include <gtest/gtest.h>
#include <lynx/lynx.h>
#include <torch/Torch.h>

namespace torch
{
namespace testing
{

TEST(LightCostFunction, Gradient)
{
  Scene scene;

  std::shared_ptr<Sphere> geometry;
  geometry = scene.CreateSphere();
  geometry->SetScale(1, 1, 1);

  std::shared_ptr<MatteMaterial> material;
  material = scene.CreateMatteMaterial();
  material->SetAlbedo(0.8, 0.2, 0.2);

  std::shared_ptr<Primitive> primitive;
  primitive = scene.CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  primitive->SetPosition(0, 0, 1);
  scene.Add(primitive);

  std::shared_ptr<EnvironmentLight> light;
  light = scene.CreateEnvironmentLight();
  light->SetRowCount(6);
  light->SetRadiance(0.5, 0.5, 0.5);
  scene.Add(light);

  std::shared_ptr<Camera> camera;
  camera = scene.CreateCamera();
  camera->SetOrientation(0, 0, 0);
  camera->SetPosition(0, 0, 0);
  camera->SetImageSize(80, 60);
  camera->SetFocalLength(40, 40);
  camera->SetCenterPoint(40, 30);
  camera->SetSampleCount(64);

  std::shared_ptr<Image> image;
  image = std::make_shared<Image>();
  camera->Capture(*image);

  std::shared_ptr<Keyframe> keyframe;
  keyframe = std::make_shared<Keyframe>(camera, image);

  lynx::Problem problem;

  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  float* values = reinterpret_cast<float*>(pointer);

  problem.AddParameterBlock(values, 3 * light->GetDirectionCount());
  problem.SetLowerBound(values, 0.0f);

  torch::LightCostFunction* costFunction;
  costFunction = new torch::LightCostFunction(light);
  costFunction->AddKeyframe(keyframe);
  problem.AddResidualBlock(costFunction, nullptr, values);

  problem.CheckGradients();
}

TEST(LightCostFunction, Optimization)
{
  Scene scene;

  std::shared_ptr<Sphere> geometry;
  geometry = scene.CreateSphere();
  geometry->SetScale(1, 1, 1);

  std::shared_ptr<MatteMaterial> material;
  material = scene.CreateMatteMaterial();
  material->SetAlbedo(0.8, 0.0, 0.0);

  std::shared_ptr<Primitive> primitive;
  primitive = scene.CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  primitive->SetPosition(0, 0, 1);
  scene.Add(primitive);

  std::shared_ptr<EnvironmentLight> light;
  light = scene.CreateEnvironmentLight();
  light->SetRowCount(3);
  light->SetRadiance(0.01, 0.00, 0.00);
  light->SetRadiance(0, Spectrum::FromRGB(5.0, 0.0, 0.0));
  scene.Add(light);

  std::shared_ptr<Camera> camera;
  camera = scene.CreateCamera();
  camera->SetOrientation(0, 0, 0);
  camera->SetPosition(0, 0, 0);
  camera->SetImageSize(160, 120);
  camera->SetFocalLength(80, 80);
  camera->SetCenterPoint(80, 60);
  camera->SetSampleCount(64);

  std::shared_ptr<Image> image;
  image = std::make_shared<Image>();
  camera->Capture(*image);

  light->SetRadiance(0.01, 0.00, 0.00);
  light->GetContext()->Compile();

  std::shared_ptr<Keyframe> keyframe;
  keyframe = std::make_shared<Keyframe>(camera, image);

  lynx::Problem problem;

  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  float* values = reinterpret_cast<float*>(pointer);

  problem.AddParameterBlock(values, 3 * light->GetDirectionCount());
  problem.SetLowerBound(values, 0.0f);

  torch::LightCostFunction* costFunction;
  costFunction = new torch::LightCostFunction(light);
  costFunction->AddKeyframe(keyframe);
  problem.AddResidualBlock(costFunction, nullptr, values);

  lynx::Solver::Summary summary;
  lynx::Solver solver(&problem);
  solver.Solve(&summary);

  std::cout << summary.BriefReport() << std::endl;

  ASSERT_TRUE(summary.solutionUsable && summary.finalCost < 0.1 &&
      !std::isnan(summary.finalCost) && !std::isinf(summary.finalCost));
}

} // namespace testing

} // namespace torch