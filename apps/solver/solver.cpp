#include <iostream>
#include <lynx/lynx.h>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  std::cout << "Starting..." << std::endl;

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
  camera->SetImageSize(80, 40);
  camera->SetFocalLength(40, 40);
  camera->SetCenterPoint(40, 30);
  camera->SetSampleCount(64);

  std::cout << "Rendering reference image..." << std::endl;

  std::shared_ptr<Image> image;
  image = std::make_shared<Image>();
  camera->Capture(*image);

  std::cout << "Creating image mask..." << std::endl;

  std::shared_ptr<Keyframe> keyframe;
  keyframe = std::make_shared<Keyframe>(camera, image);

  lynx::Problem problem;

  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  float* values = reinterpret_cast<float*>(pointer);

  problem.AddParameterBlock(values, 3 * light->GetDirectionCount());
  problem.SetLowerBound(values, 0.0f);

  std::cout << "Creating cost function..." << std::endl;

  torch::LightCostFunction* costFunction;
  costFunction = new torch::LightCostFunction(light);
  costFunction->AddKeyframe(keyframe);
  problem.AddResidualBlock(costFunction, nullptr, values);

  std::cout << "Checking gradient..." << std::endl;

  problem.CheckGradients();

  std::cout << "Success" << std::endl;
  return 0;
}