#include <iostream>
#include <lynx/lynx.h>
#include <torch/Torch.h>

using namespace torch;

uint iteration = 0;

void Solve(std::vector<Spectrum>& radiance)
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
  light->SetRowCount(21);
  light->SetRadiance(0.001, 0.001, 0.001);
  light->SetRadiance(0, Spectrum::FromRGB(5.0, 5.0, 5.0));
  scene.Add(light);

  std::shared_ptr<Camera> camera;
  camera = scene.CreateCamera();
  camera->SetOrientation(0, 0, 0);
  camera->SetPosition(0, 0, 0);
  camera->SetImageSize(160, 120);
  camera->SetFocalLength(80, 80);
  camera->SetCenterPoint(80, 60);
  camera->SetSampleCount(16);

  std::cout << "Rendering reference image..." << std::endl;

  std::shared_ptr<Image> image;
  image = std::make_shared<Image>();
  camera->Capture(*image);
  image->Save("reference.png");

  light->SetRadiance(radiance);
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

  lynx::Solver::Options options;
  options.maxIterations = 10000;

  lynx::Solver solver(&problem);
  solver.Configure(options);

  lynx::Solver::Summary summary;
  solver.Solve(&summary);

  std::cout << summary.BriefReport() << std::endl;

  std::cout << "Rendering final estimate..." << std::endl;

  camera->Capture(*image);
  image->Save("final" + std::to_string(iteration) + ".png");

  LYNX_CHECK_CUDA(cudaMemcpy(radiance.data(), values,
      sizeof(float) * costFunction->GetParameterCount(),
      cudaMemcpyDeviceToHost))

  ++iteration;
}

int main(int argc, char** argv)
{
  std::vector<Spectrum> radiance(522);
  Spectrum initial = Spectrum::FromRGB(0.001, 0.001, 0.001);
  std::fill(radiance.begin(), radiance.end(), initial);
  Solve(radiance);
  Solve(radiance);
  return 0;
}