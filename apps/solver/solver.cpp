#include <ctime>
#include <iostream>
#include <lynx/lynx.h>
#include <torch/Torch.h>

using namespace torch;

uint iteration = 0;
std::shared_ptr<Scene> scene;
std::shared_ptr<EnvironmentLight> light;
std::shared_ptr<MatteMaterial> material;
std::shared_ptr<Mesh> mesh;

std::vector<std::shared_ptr<Camera>> cameras;
std::vector<std::shared_ptr<Keyframe>> keyframes;

lynx::Problem* lightProblem;
LightCostFunction* lightCostFunction;
float* lightValues;

lynx::Problem* albedoProblem;
AlbedoCostFunction* albedoCostFunction;
ReflectanceCostFunction* refCostFunction;
DarkenCostFunction* darkenCostFunction;
float* albedoValues;

void BuildScene()
{
  std::cout << "Building scene..." << std::endl;

  scene = std::make_shared<Scene>();

  // mesh = scene->CreateMesh("../shark.ply");
  // std::shared_ptr<Material> temp = scene->CreateMaterial("../shark.ply");
  // mesh = scene->CreateMesh("/home/mike/Downloads/initial_mesh.ply");
  // std::shared_ptr<Material> temp = scene->CreateMaterial("/home/mike/Downloads/initial_mesh.ply");
  // material = std::static_pointer_cast<MatteMaterial>(temp);

  // material = scene->CreateMatteMaterial();
  // std::vector<Spectrum> albedos(mesh->GetVertexCount());
  // std::fill(albedos.begin(), albedos.end(), Spectrum::FromRGB(0.5, 0.5, 0.5));
  // material->SetAlbedos(albedos);

  // mesh = scene->CreateMesh("/home/mike/Desktop/Datasets/PurpleRoomCorner02/painted_mesh.ply");
  // std::shared_ptr<Material> temp = scene->CreateMaterial("/home/mike/Desktop/Datasets/PurpleRoomCorner02/painted_mesh.ply");
  mesh = scene->CreateMesh("/home/mike/Desktop/out_mesh_small.ply");
  std::shared_ptr<Material> temp = scene->CreateMaterial("/home/mike/Desktop/out_mesh_small.ply");
  material = std::static_pointer_cast<MatteMaterial>(temp);

  std::shared_ptr<Primitive> primitive;
  primitive = scene->CreatePrimitive();
  primitive->SetGeometry(mesh);
  primitive->SetMaterial(material);
  scene->Add(primitive);

  light = scene->CreateEnvironmentLight();
  light->SetRowCount(21);
  // light->SetRadiance(0.001, 0.001, 0.001);
  // light->SetRadiance(67 , Spectrum::FromRGB(2.75, 2.75, 2.75));
  light->SetRadiance(1E-5, 1E-5, 1E-5);
  scene->Add(light);

  cameras.resize(1);

  const float scale = 4;
  cameras[0] = scene->CreateCamera();
  cameras[0]->SetImageSize(640 / scale, 480 / scale);
  cameras[0]->SetFocalLength(535.7239 / scale, 536.2900 / scale);
  cameras[0]->SetCenterPoint(320.2685 / scale, 240.2924 / scale);
  // cameras[0]->SetOrientation(-0.00242877, -0.0601033, -0.0349038, 0.997579);
  // cameras[0]->SetPosition(0.0761918, 0.124095, 0.00571185);
  cameras[0]->SetOrientation(-0.000125662, 0.0251924, -0.00142194, 0.999682);
  cameras[0]->SetPosition(-0.00808962, 0.0115375, -0.0037175);
  cameras[0]->SetSampleCount(5);

  // const float scale = 1;
  // cameras[0] = scene->CreateCamera();
  // cameras[0]->SetOrientation(0.00362368, 0.00285626, -0.00825128);
  // cameras[0]->SetPosition(0.0263875, 0.000004579, 0.0032462);
  // cameras[0]->SetOrientation(2.50782, 2.69967, -2.79901);
  // cameras[0]->SetPosition(-0.470782, -0.475601, 0.34118);
  // cameras[0]->SetImageSize(640 / scale, 480 / scale);
  // cameras[0]->SetFocalLength(535.7239 / scale, 536.2900 / scale);
  // cameras[0]->SetCenterPoint(320.2685 / scale, 240.2924 / scale);
  // cameras[0]->SetSampleCount(4);

  // cameras.resize(3);

  // cameras[0] = scene->CreateCamera();
  // cameras[0]->SetOrientation(0, 0, 0);
  // cameras[0]->SetPosition(0, 0, 0.5);
  // cameras[0]->SetImageSize(160, 120);
  // cameras[0]->SetFocalLength(80, 80);
  // cameras[0]->SetCenterPoint(80, 60);
  // cameras[0]->SetSampleCount(1);

  // cameras[1] = scene->CreateCamera();
  // cameras[1]->SetOrientation(-M_PIf / 5, -M_PIf / 4, 0);
  // cameras[1]->SetPosition(0.6, -0.2, 0.8);
  // cameras[1]->SetImageSize(160, 120);
  // cameras[1]->SetFocalLength(80, 80);
  // cameras[1]->SetCenterPoint(80, 60);
  // cameras[1]->SetSampleCount(1);

  // cameras[2] = scene->CreateCamera();
  // cameras[2]->SetOrientation(-M_PIf / 3.5, 0, 0);
  // cameras[2]->SetPosition(0.2, -0.6, 0.75);
  // cameras[2]->SetImageSize(160, 120);
  // cameras[2]->SetFocalLength(80, 80);
  // cameras[2]->SetCenterPoint(80, 60);
  // cameras[2]->SetSampleCount(1);

  std::cout << "Rendering reference image..." << std::endl;

  keyframes.resize(cameras.size());

  for (size_t i = 0; i < cameras.size(); ++i)
  {
    std::shared_ptr<Camera> camera = cameras[i];

    std::shared_ptr<Image> image;
    image = std::make_shared<Image>();
    camera->Capture(*image);
    // image->Load("/home/mike/Downloads/reference2.png");
    // image->Save("reference_" + std::to_string(i) + ".png");
    // image->Load("/home/mike/Desktop/Datasets/PurpleRoomCorner02/clean/image_rgb_00160.ppm");
    // image->Load("/home/mike/Desktop/image_rgb_00160.ppm");
    image->Load("/home/mike/Desktop/image_rgb_00019.ppm");

    keyframes[i] = std::make_shared<Keyframe>(camera, image);
  }

  // std::cout << "Baking albedos..." << std::endl;

  // AlbedoBaker baker(scene);
  // baker.SetSampleCount(4);
  // baker.Bake(material, mesh);

  // std::cout << "Saving baked mesh..." << std::endl;

  // material->LoadAlbedos();
  // MeshWriter writer(mesh, material);
  // writer.Write("baked_mesh.ply");

  // light->SetRadiance(1E-5, 1E-5, 1E-5);
}

void CreateLightProblem()
{
  std::cout << "Creating light problem..." << std::endl;

  const size_t paramCount = 3 * light->GetDirectionCount();
  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  lightValues = reinterpret_cast<float*>(pointer);

  lightProblem = new lynx::Problem();
  lightProblem->AddParameterBlock(lightValues, paramCount);
  lightProblem->SetLowerBound(lightValues, 0.0f);

  lightCostFunction = new torch::LightCostFunction(light);

  for (std::shared_ptr<Keyframe> keyframe : keyframes)
  {
    lightCostFunction->AddKeyframe(keyframe);
  }

  lightProblem->AddResidualBlock(lightCostFunction, nullptr, lightValues);

  ActivationCostFunction* actCostFunction;
  actCostFunction = new ActivationCostFunction(light);
  actCostFunction->SetBias(1.0);
  actCostFunction->SetInnerScale(1.0);
  actCostFunction->SetOuterScale(1.0);
  lightProblem->AddResidualBlock(actCostFunction, nullptr, lightValues);
}

void CreateAlbedoProblem()
{
  std::cout << "Creating albedo problem..." << std::endl;

  const size_t paramCount = 3 * material->GetAlbedoCount();
  optix::Buffer buffer = material->GetAlbedoBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  albedoValues = reinterpret_cast<float*>(pointer);

  albedoProblem = new lynx::Problem();
  albedoProblem->AddParameterBlock(albedoValues, paramCount);
  albedoProblem->SetLowerBound(albedoValues, 0.0f);
  albedoProblem->SetUpperBound(albedoValues, 1.0f);

  albedoCostFunction = new torch::AlbedoCostFunction(material, mesh);

  for (std::shared_ptr<Keyframe> keyframe : keyframes)
  {
    albedoCostFunction->AddKeyframe(keyframe);
  }

  albedoProblem->AddResidualBlock(albedoCostFunction, nullptr, albedoValues);

  refCostFunction = new ReflectanceCostFunction(material, mesh);
  refCostFunction->SetChromaticityThreshold(0.995);
  refCostFunction->SetWeight(0.4);
  albedoProblem->AddResidualBlock(refCostFunction, nullptr, albedoValues);

  darkenCostFunction = new DarkenCostFunction(paramCount);
  // darkenCostFunction->SetWeight(0.1);
  // albedoProblem->AddResidualBlock(darkenCostFunction, nullptr, albedoValues);
}

void SolveLightProblem()
{
  std::cout << "Solving light problem..." << std::endl;

  lightCostFunction->ClearJacobian();

  lynx::Solver::Options options;
  options.maxIterations = 50000;
  options.minCostChangeRate = 1E-5;
  options.verbose = true;

  lynx::Solver solver(lightProblem);
  solver.Configure(options);

  lynx::Solver::Summary summary;
  solver.Solve(&summary);

  std::cout << summary.BriefReport() << std::endl;

  const size_t count = lightProblem->GetParameterBlockSize(0);
  std::vector<Spectrum> radiance(count / 3);

  const size_t bytes = sizeof(float) * count;
  const cudaMemcpyKind type = cudaMemcpyDeviceToHost;
  LYNX_CHECK_CUDA(cudaMemcpy(radiance.data(), lightValues, bytes, type));
  light->SetRadiance(radiance);

  std::cout << "Rendering new light estimate..." << std::endl;

  Image image;

  for (size_t i = 0; i < cameras.size(); ++i)
  {
    cameras[i]->CaptureLighting(image);

    image.Save("light_estimate_" + std::to_string(i) + "_" +
        std::to_string(iteration) + ".png");
  }
}

void SolveAlbedoProblem()
{
  std::cout << "Solving albedo problem..." << std::endl;

  albedoCostFunction->ClearJacobian();
  darkenCostFunction->SetValues(albedoValues);

  lynx::Solver::Options options;
  options.maxIterations = 50000;
  options.verbose = true;

  lynx::Solver solver(albedoProblem);
  solver.Configure(options);

  lynx::Solver::Summary summary;
  solver.Solve(&summary);

  std::cout << summary.BriefReport() << std::endl;

  const size_t count = albedoProblem->GetParameterBlockSize(0);
  std::vector<Spectrum> albedos(count / 3);

  const size_t bytes = sizeof(float) * count;
  const cudaMemcpyKind type = cudaMemcpyDeviceToHost;
  LYNX_CHECK_CUDA(cudaMemcpy(albedos.data(), albedoValues, bytes, type));
  material->SetAlbedos(albedos);

  std::cout << "Rendering new albedo estimate..." << std::endl;

  Image image;

  for (size_t i = 0; i < cameras.size(); ++i)
  {
    cameras[i]->CaptureAlbedo(image);

    image.Save("albedo_estimate_" + std::to_string(i) + "_" +
        std::to_string(iteration) + ".png");
  }
}

void SolveProblem()
{
  BuildScene();
  CreateAlbedoProblem();
  CreateLightProblem();

  int i = 0;

  for (iteration = 0; iteration < 20; ++iteration)
  {
    std::cout << "Starting iteration " << iteration << "..." << std::endl;
    SolveAlbedoProblem();
    SolveLightProblem();

    if (iteration == 0)
    {
      ++iteration; SolveLightProblem();
      ++iteration; SolveLightProblem();
      ++iteration; SolveLightProblem();
    }

    refCostFunction->SetWeight(2.0);

    if (iteration % 5 == 0)
    {
      std::cout << "Writing final mesh estimate..." << std::endl;
      MeshWriter writer(mesh, material);
      writer.Write("mesh_estimate_" + std::to_string(iteration) + ".ply");
    }

    albedoProblem->AddResidualBlock(albedoCostFunction, nullptr, albedoValues);

    ++i;
  }

  std::cout << "Writing final mesh estimate..." << std::endl;

  MeshWriter writer(mesh, material);
  writer.Write("mesh_estimate.ply");
}

int main(int argc, char** argv)
{
  const clock_t start = clock();
  SolveProblem();
  const clock_t stop = clock();
  const double time = double(stop - start) / CLOCKS_PER_SEC;
  std::cout << "Elapsed time: " << time << std::endl;
  return 0;
}
