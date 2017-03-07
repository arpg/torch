#include <gtest/gtest.h>
#include <torch/Torch.h>

namespace torch
{
namespace testing
{

TEST(MeshCostFunction, Gradient)
{
  std::vector<Spectrum> albedos;
  std::vector<Point> vertices;
  std::vector<Normal> normals;
  std::vector<uint3> faces;

  Spectrum albedo = Spectrum::FromRGB(0.8, 0.4, 0.4);

  for (int x = -5; x < 5; ++x)
  {
    for (int y = -5; y < 5; ++y)
    {
      float xf = 0.5f * x;
      float yf = 0.5f * y;
      float zf = 2.0f - sqrtf(xf*xf + yf*yf) / 5.0;
      Point p(xf, yf, zf);
      Vector v = p - Point(0, 0, 5);

      vertices.push_back(p);
      normals.push_back(Normal(v));
      albedos.push_back(albedo);

      if (x < 4 && y < 4)
      {
        unsigned int c  = (x + 5) * 11 + (y + 5);
        unsigned int s  = (x + 5) * 11 + (y + 6);
        unsigned int e  = (x + 6) * 11 + (y + 5);
        unsigned int se = (x + 6) * 11 + (y + 6);
        faces.push_back(make_uint3(c, s, se));
        faces.push_back(make_uint3(c, se, e));
      }
    }
  }

  std::shared_ptr<Scene> scene;
  scene = std::make_shared<Scene>();

  std::shared_ptr<Mesh> geometry;
  geometry = scene->CreateMesh();
  geometry->SetVertices(vertices);
  geometry->SetNormals(normals);
  geometry->SetFaces(faces);

  std::shared_ptr<MatteMaterial> material;
  material = scene->CreateMatteMaterial();
  material->SetAlbedos(albedos);

  std::shared_ptr<Primitive> primitive;
  primitive = scene->CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  primitive->SetPosition(0, 0, 1);
  scene->Add(primitive);

  std::shared_ptr<VoxelLight> light;
  light = scene->CreateVoxelLight();
  light->SetPosition(0, 0, -2.5);
  light->SetDimensions(5, 5, 1);
  light->SetVoxelSize(2.0);
  light->SetRadiance(0.50, 0.50, 0.50);
  light->SetRadiance(0, Spectrum::FromRGB(10.3, 10.3, 10.3));
  scene->Add(light);

  AlbedoBaker baker(scene);
  baker.SetSampleCount(10);
  baker.Bake(material, geometry);

  light->SetRadiance(0, Spectrum::FromRGB(8.3, 8.3, 8.3));
  light->GetContext()->Compile();

  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  float* values = reinterpret_cast<float*>(pointer);

  lynx::Problem problem;

  problem.AddParameterBlock(values, 3 * light->GetVoxelCount());
  problem.SetLowerBound(values, 0.0f);

  MeshCostFunction* costFunction;
  costFunction = new MeshCostFunction(light, geometry, material);
  costFunction->SetMaxNeighborCount(5);
  costFunction->SetMaxNeighborDistance(1.5f);
  costFunction->SetSimilarityThreshold(0.0f);
  costFunction->SetLightSampleCount(1000);
  problem.AddResidualBlock(costFunction, nullptr, values);

  problem.CheckGradients();
}

TEST(MeshCostFunction, Optimization)
{
  std::vector<Spectrum> albedos;
  std::vector<Point> vertices;
  std::vector<Normal> normals;
  std::vector<uint3> faces;

  Spectrum albedo = Spectrum::FromRGB(0.8, 0.4, 0.4);

  for (int x = -5; x < 6; ++x)
  {
    for (int y = -5; y < 6; ++y)
    {
      float xf = 0.5f * x;
      float yf = 0.5f * y;
      float zf = 1.0f + sqrtf(xf*xf + yf*yf) / 5.0;
      Point p(xf, yf, zf);
      Vector v = p - Point(0, 0, 5);

      vertices.push_back(p);
      normals.push_back(Normal(v));
      albedos.push_back(albedo);

      if (x < 4 && y < 4)
      {
        unsigned int c  = (x + 5) * 11 + (y + 5);
        unsigned int s  = (x + 5) * 11 + (y + 6);
        unsigned int e  = (x + 6) * 11 + (y + 5);
        unsigned int se = (x + 6) * 11 + (y + 6);
        faces.push_back(make_uint3(c, s, se));
        faces.push_back(make_uint3(c, se, e));
      }
    }
  }

  std::shared_ptr<Scene> scene;
  scene = std::make_shared<Scene>();

  std::shared_ptr<Mesh> geometry;
  geometry = scene->CreateMesh();
  geometry->SetVertices(vertices);
  geometry->SetNormals(normals);
  geometry->SetFaces(faces);

  std::shared_ptr<MatteMaterial> material;
  material = scene->CreateMatteMaterial();
  material->SetAlbedos(albedos);

  std::shared_ptr<Primitive> primitive;
  primitive = scene->CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  primitive->SetPosition(0, 0, 1);
  scene->Add(primitive);

  std::shared_ptr<VoxelLight> light;
  light = scene->CreateVoxelLight();
  light->SetPosition(0, 0, -2.5);
  light->SetDimensions(5, 5, 1);
  light->SetVoxelSize(2.0);
  light->SetRadiance(0.05, 0.05, 0.05);
  light->SetRadiance(0, Spectrum::FromRGB(30, 30, 30));
  scene->Add(light);

  std::shared_ptr<Camera> camera;
  camera = scene->CreateCamera();
  camera->SetPosition(0, 0, -2);
  camera->SetImageSize(640, 480);
  camera->SetFocalLength(320, 320);
  camera->SetCenterPoint(320, 240);
  camera->SetSampleCount(5);

  Image image;
  camera->Capture(image);
  image.Save("test_image1.png");

  camera->CaptureAlbedo(image);
  image.Save("test_image1_5.png");

  AlbedoBaker baker(scene);
  baker.SetSampleCount(10);
  baker.Bake(material, geometry);

  // light->SetRadiance(10.1, 10.1, 10.1);
  light->SetRadiance(0, Spectrum::FromRGB(20, 20, 20));

  camera->Capture(image);
  image.Save("test_image2.png");

  camera->CaptureAlbedo(image);
  image.Save("test_image2_5.png");

  light->GetContext()->Compile();

  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  float* values = reinterpret_cast<float*>(pointer);

  lynx::Problem problem;

  problem.AddParameterBlock(values, 3 * light->GetVoxelCount());
  problem.SetLowerBound(values, 0.0f);

  MeshCostFunction* costFunction;
  costFunction = new MeshCostFunction(light, geometry, material);
  costFunction->SetMaxNeighborCount(5);
  costFunction->SetMaxNeighborDistance(1.5f);
  costFunction->SetSimilarityThreshold(0.0f);
  costFunction->SetLightSampleCount(1000);
  problem.AddResidualBlock(costFunction, nullptr, values);

  lynx::Solver::Options options;
  options.maxIterations = 10000;
  options.minCostChangeRate = 1E-9;

  lynx::Solver solver(&problem);
  solver.Configure(options);

  lynx::Solver::Summary summary;
  solver.Solve(&summary);

  std::cout << summary.BriefReport() << std::endl;

  camera->CaptureLighting(image);
  image.Save("test_image3.png");

  ShadingRemover remover(geometry, material);
  remover.SetSampleCount(1000);
  remover.Remove();

  camera->CaptureAlbedo(image);
  image.Save("test_image3_5.png");

  std::vector<float> rvalues(3 * light->GetVoxelCount());
  optix::Buffer radiance = light->GetRadianceBuffer();
  float* device = reinterpret_cast<float*>(radiance->map());
  std::copy(device, device + rvalues.size(), rvalues.data());
  radiance->unmap();

  for (size_t i = 0; i < light->GetVoxelCount(); ++i)
  {
    std::cout << rvalues[3 * i + 0] << " ";
    std::cout << rvalues[3 * i + 1] << " ";
    std::cout << rvalues[3 * i + 2] << std::endl;
  }

  // ASSERT_TRUE(summary.solutionUsable && summary.finalCost < 1E-6);
}

} // namespace testing

} // namespace torch