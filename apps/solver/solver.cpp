#include <iostream>
#include <lynx/lynx.h>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  std::cout << "Starting..." << std::endl;

  Scene scene;

  std::shared_ptr<Mesh> geometry;
  geometry = scene.CreateMesh("../shark.ply");
  geometry->SetScale(1, 1, 1);

  std::shared_ptr<Material> material;
  material = scene.CreateMaterial("../shark.ply");

  std::shared_ptr<Primitive> primitive;
  primitive = scene.CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  primitive->SetPosition(0, 0, -0.5);
  scene.Add(primitive);

  std::shared_ptr<EnvironmentLight> light;
  light = scene.CreateEnvironmentLight();
  light->SetRowCount(3);
  light->SetRadiance(0.01, 0.01, 0.01);
  light->SetRadiance(0, Spectrum::FromRGB(5.0, 5.0, 5.0));
  scene.Add(light);

  std::shared_ptr<Camera> camera;
  camera = scene.CreateCamera();
  camera->SetOrientation(0, 0, 0);
  camera->SetPosition(0, 0, 0);
  camera->SetImageSize(80, 60);
  camera->SetFocalLength(40, 40);
  camera->SetCenterPoint(40, 30);
  camera->SetSampleCount(8);

  std::cout << "Rendering reference image..." << std::endl;

  std::shared_ptr<Image> image;
  image = std::make_shared<Image>();
  camera->Capture(*image);
  image->Save("reference.png");

  light->SetRadiance(0.01, 0.01, 0.01);
  light->GetContext()->Compile();

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

  std::cout << "Solving problem..." << std::endl;

  lynx::Solver::Summary summary;
  lynx::Solver solver(&problem);
  solver.Solve(&summary);
  std::cout << summary.BriefReport() << std::endl;

  std::cout << "Rendering final estimate..." << std::endl;

  camera->Capture(*image);
  image->Save("final.png");

  std::cout << "Success" << std::endl;
  return 0;
}