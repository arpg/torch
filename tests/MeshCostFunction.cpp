#include <gtest/gtest.h>
#include <torch/Torch.h>

namespace torch
{
namespace testing
{

TEST(MeshCostFunction, General)
{
  std::shared_ptr<Scene> scene;
  scene = std::make_shared<Scene>();

  std::vector<Spectrum> albedos;
  albedos.push_back(Spectrum::FromRGB(1, 1, 1));
  albedos.push_back(Spectrum::FromRGB(1, 1, 1));
  albedos.push_back(Spectrum::FromRGB(1, 1, 1));

  std::vector<Point> vertices;
  vertices.push_back(Point(0, 0, 0));
  vertices.push_back(Point(0, 1, 0));
  vertices.push_back(Point(1, 1, 0));

  std::vector<Normal> normals;
  normals.push_back(Normal(0, 0, -1));
  normals.push_back(Normal(0, 0, -1));
  normals.push_back(Normal(0, 0, -1));

  std::vector<uint3> faces;
  faces.push_back(make_uint3(0, 1, 2));

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
  light->SetPosition(0, 0, -1);
  light->SetDimensions(2, 2, 1);
  light->SetVoxelSize(0.1);
  light->SetRadiance(0.1, 0.1, 0.1);
  light->SetRadiance(0, Spectrum::FromRGB(5, 5, 5));
  scene->Add(light);

  AlbedoBaker baker(scene);
  // baker.SetSampleCount(10);
  baker.SetSampleCount(128);
  baker.Bake(material, geometry);

  material->LoadAlbedos();
  std::vector<Spectrum> new_colors;
  material->GetAlbedos(new_colors);

  for (const Spectrum& color : new_colors)
  {
    const Vector rgb = color.GetRGB();
    std::cout << "Color: " << rgb.x << " " << rgb.y << " " << rgb.z << std::endl;
  }

  light->SetRadiance(0.1, 0.1, 0.1);
  light->GetContext()->Compile();

  MeshCostFunction* costFunction;
  costFunction = new MeshCostFunction(light, geometry, material);
  costFunction->SetMaxNeighborCount(3);
  costFunction->SetMaxNeighborDistance(2.0f);
  costFunction->SetSimilarityThreshold(0.0f);
  costFunction->SetLightSampleCount(2048);

  // costFunction->CreateJacobianMatrix();

  // ASSERT_EQ(9, costFunction->GetResidualCount());

  // float* parameters;
  // LYNX_CHECK_CUDA(cudaMalloc(&parameters, sizeof(float) * 3 * light->GetVoxelCount()));
  // lynx::Fill(parameters, 0.1, 3 * light->GetVoxelCount());

  // float* residuals;
  // LYNX_CHECK_CUDA(cudaMalloc(&residuals, sizeof(float) * 9));
  // lynx::SetZeros(residuals, 9);

  // float* gradient;
  // LYNX_CHECK_CUDA(cudaMalloc(&gradient, sizeof(float) * 3 * light->GetVoxelCount()));
  // lynx::SetZeros(gradient, 3 * light->GetVoxelCount());

  // costFunction->Evaluate(&parameters, residuals, gradient);

  // double cost = 0;

  // for (size_t i = 0; i < 9; ++i)
  // {
  //   const float value = lynx::Get(&residuals[i]);
  //   std::cout << "Residual " << i << ": " << value << std::endl;
  //   cost += value * value;
  // }

  // std::cout << std::endl;
  // std::cout << "Cost: " << cost << std::endl;
  // std::cout << std::endl;

  // for (size_t i = 0; i < 3 * light->GetVoxelCount(); ++i)
  // {
  //   std::cout << "Gradient " << i << ": " << lynx::Get(&gradient[i]) << std::endl;
  // }

  // std::cout << std::endl;
  // std::cout << std::endl;

  // lynx::Set(&parameters[0], 5);
  // lynx::Set(&parameters[1], 5);
  // lynx::Set(&parameters[2], 5);

  // costFunction->Evaluate(&parameters, residuals, gradient);

  // cost = 0;

  // for (size_t i = 0; i < 9; ++i)
  // {
  //   const float value = lynx::Get(&residuals[i]);
  //   std::cout << "Residual " << i << ": " << value << std::endl;
  //   cost += value * value;
  // }

  // std::cout << std::endl;
  // std::cout << "Cost: " << cost << std::endl;
  // std::cout << std::endl;

  // for (size_t i = 0; i < 3 * light->GetVoxelCount(); ++i)
  // {
  //   std::cout << "Gradient " << i << ": " << lynx::Get(&gradient[i]) << std::endl;
  // }




  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  float* values = reinterpret_cast<float*>(pointer);

  lynx::Problem problem;

  problem.AddParameterBlock(values, 3 * light->GetVoxelCount());
  problem.SetLowerBound(values, 0.0f);

  problem.AddResidualBlock(costFunction, nullptr, values);

  // problem.CheckGradients();

  lynx::Solver::Options options;
  options.maxIterations = 10000;
  options.minCostChangeRate = 1E-20;

  lynx::Solver solver(&problem);
  solver.Configure(options);

  lynx::Solver::Summary summary;
  solver.Solve(&summary);

  std::cout << summary.BriefReport() << std::endl;
  std::cout << std::endl;

  for (size_t i = 0; i < light->GetVoxelCount(); ++i)
  {
    std::cout << "Voxel Value " << i << ": ";
    std::cout << lynx::Get(values + 3 * i + 0) << " ";
    std::cout << lynx::Get(values + 3 * i + 1) << " ";
    std::cout << lynx::Get(values + 3 * i + 2) << std::endl;
  }

  std::cout << std::endl;
}
















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
  light->SetDimensions(2, 2, 1);
  light->SetVoxelSize(2.0);
  light->SetRadiance(0.50, 0.50, 0.50);
  light->SetRadiance(0, Spectrum::FromRGB(10.3, 10.3, 10.3));
  scene->Add(light);

  AlbedoBaker baker(scene);
  baker.SetSampleCount(10);
  baker.Bake(material, geometry);

  // light->SetRadiance(0, Spectrum::FromRGB(8.3, 8.3, 8.3));
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
  costFunction->SetMaxNeighborDistance(1.0f);
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
  Spectrum albedo2 = Spectrum::FromRGB(0.4, 0.4, 0.8);

  const float size = 5.0f;
  const unsigned int resolution = 50;
  const float stepSize = size / resolution;
  const float origin = -size / 2.0;

  for (unsigned int x = 0; x <= resolution; ++x)
  {
    for (unsigned int y = 0; y <= resolution; ++y)
    {
      float xf = origin + stepSize * x;
      float yf = origin + stepSize * y;
      float zf = 2.0f + sqrtf(xf*xf + yf*yf) / 5.0;
      Point p(xf, yf, zf);
      Vector n = p - Point(0, 0, 5);

      vertices.push_back(p);
      normals.push_back(Normal(n));
      albedos.push_back((x < resolution / 2) ? albedo : albedo2);

      if (x < resolution && y < resolution)
      {
        unsigned int c  = (x + 0) * (resolution + 1) + (y + 0);
        unsigned int s  = (x + 0) * (resolution + 1) + (y + 1);
        unsigned int e  = (x + 1) * (resolution + 1) + (y + 0);
        unsigned int se = (x + 1) * (resolution + 1) + (y + 1);
        faces.push_back(make_uint3(c, s, se));
        faces.push_back(make_uint3(c, se, e));
      }
    }
  }

  size_t ii = vertices.size();
  vertices.push_back(Point(-0.5, -0.5, 0.8));
  vertices.push_back(Point(-0.5, -1.5, 1.0));
  vertices.push_back(Point(-1.5, -1.5, 1.0));
  normals.push_back(Normal(0.0, 0.0, -1.0));
  normals.push_back(Normal(0.0, 0.0, -1.0));
  normals.push_back(Normal(0.0, 0.0, -1.0));
  faces.push_back(make_uint3(ii + 0, ii + 1, ii + 2));
  albedos.push_back(Spectrum::FromRGB(0.4, 0.8, 0.4));
  albedos.push_back(Spectrum::FromRGB(0.4, 0.8, 0.4));
  albedos.push_back(Spectrum::FromRGB(0.4, 0.8, 0.4));

  ii = vertices.size();

  std::shared_ptr<Scene> scene;
  scene = std::make_shared<Scene>();
  vertices.push_back(Point(-0.5, 0.2, 1.0));
  vertices.push_back(Point(1.0, 0.0, 1.2));
  vertices.push_back(Point(1.0, 1.0, 1.2));
  normals.push_back(Normal(0.0, 0.0, -1.0));
  normals.push_back(Normal(0.0, 0.0, -1.0));
  normals.push_back(Normal(0.0, 0.0, -1.0));
  faces.push_back(make_uint3(ii + 0, ii + 1, ii + 2));
  albedos.push_back(Spectrum::FromRGB(0.4, 0.8, 0.4));
  albedos.push_back(Spectrum::FromRGB(0.4, 0.8, 0.4));
  albedos.push_back(Spectrum::FromRGB(0.4, 0.8, 0.4));

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
  primitive->SetPosition(0, 0, 0);
  scene->Add(primitive);

  std::shared_ptr<VoxelLight> light;
  light = scene->CreateVoxelLight();
  light->SetPosition(0, 0, -3);
  // light->SetDimensions(9, 9, 9);
  light->SetDimensions(3, 3, 3);
  light->SetVoxelSize(1.0);
  light->SetRadiance(0.01, 0.01, 0.01);
  light->SetRadiance(0, Spectrum::FromRGB(100, 100, 100));
  scene->Add(light);

  std::shared_ptr<Camera> camera;
  camera = scene->CreateCamera();
  camera->SetPosition(0, 0, -2);
  camera->SetImageSize(640, 480);
  camera->SetFocalLength(320, 320);
  camera->SetCenterPoint(320, 240);
  camera->SetSampleCount(3);

  Image image;

  camera->CaptureAlbedo(image);
  image.Save("image_01_groundtruth_albedo.png");

  camera->CaptureLighting(image);
  image.Save("image_02_groundtruth_lighting.png");

  camera->Capture(image);
  image.Save("image_03_render_with_groundtruth_lighting.png");

  AlbedoBaker baker(scene);
  baker.SetSampleCount(10);
  baker.Bake(material, geometry);
  material->LoadAlbedos();

  // light->SetRadiance(0.01, 0.01, 0.01);
  light->SetRadiance(1000, 1000, 1000);
  light->GetContext()->Compile();

  // light->SetRadiance(0, Spectrum::FromRGB(10, 10, 10));
  // light->SetRadiance(24, Spectrum::FromRGB(100, 100, 100));

  camera->CaptureAlbedo(image);
  image.Save("image_04_initial_albedo.png");

  camera->CaptureLighting(image);
  image.Save("image_05_initial_lighting.png");

  camera->Capture(image);
  image.Save("image_06_render_with_initial_lighting.png");

  material->GetAlbedos(albedos);
  ShadingRemover remover2(geometry, material);
  remover2.SetSampleCount(1000);
  remover2.Remove();

  camera->CaptureAlbedo(image);
  image.Save("image_07_computed_albedos_with_initial_lighting.png");

  material->SetAlbedos(albedos);
  light->GetContext()->Compile();

  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  float* values = reinterpret_cast<float*>(pointer);

  // for (size_t i = 0; i < light->GetVoxelCount(); ++i)
  // {
  //   std::cout << "Voxel Value " << i << ": ";
  //   std::cout << lynx::Get(values + 3 * i + 0) << " ";
  //   std::cout << lynx::Get(values + 3 * i + 1) << " ";
  //   std::cout << lynx::Get(values + 3 * i + 2) << std::endl;
  // }

  lynx::Problem problem;

  problem.AddParameterBlock(values, 3 * light->GetVoxelCount());
  problem.SetLowerBound(values, 0.0f);

  MeshCostFunction* costFunction;
  costFunction = new MeshCostFunction(light, geometry, material);
  costFunction->SetMaxNeighborCount(20);
  costFunction->SetMaxNeighborDistance(4.0f);
  costFunction->SetSimilarityThreshold(0.0f);
  costFunction->SetLightSampleCount(1000);
  problem.AddResidualBlock(costFunction, nullptr, values);

  // float cost;
  // std::cout << "======== COST ========" << std::endl;
  // std::cout << "======== COST ========" << std::endl;

  // for (int i = 0; i < 1000; ++i)
  // {
  //   lynx::Set(values + 0, 100.0f / (i / 1000.0f));
  //   lynx::Set(values + 1, 100.0f / (i / 1000.0f));
  //   lynx::Set(values + 2, 100.0f / (i / 1000.0f));
  //   problem.Evaluate(&cost, nullptr, nullptr);
  //   std::cout << cost << std::endl;
  // }

  // std::cout << "======== COST ========" << std::endl;
  // std::cout << "======== COST ========" << std::endl;

  // // VoxelActivationCostFunction* actFunction;
  // // actFunction = new VoxelActivationCostFunction(light);
  // // actFunction->SetBias(1.0);
  // // actFunction->SetInnerScale(10.0);
  // // actFunction->SetOuterScale(8.0);
  // // problem.AddResidualBlock(actFunction, nullptr, values);

  lynx::Solver::Options options;
  options.maxIterations = 1000000;
  options.minCostChangeRate = 1E-6;
  options.verbose = true;

  lynx::Solver solver(&problem);
  solver.Configure(options);

  lynx::Solver::Summary summary;
  solver.Solve(&summary);

  std::cout << summary.BriefReport() << std::endl;

  std::vector<Spectrum> rvalues(light->GetVoxelCount());
  optix::Buffer radiance = light->GetRadianceBuffer();
  Spectrum* device = reinterpret_cast<Spectrum*>(radiance->map());
  std::copy(device, device + rvalues.size(), rvalues.data());
  radiance->unmap();

  light->SetRadiance(rvalues);

  camera->Capture(image);
  image.Save("image_08_render_with_final_lighting_and_initial_albedos.png");

  ShadingRemover remover(geometry, material);
  remover.SetSampleCount(1000);
  remover.Remove();

  material->LoadAlbedos();
  camera->CaptureAlbedo(image);
  image.Save("image_09_computed_albedos_with_final_lighting.png");

  const size_t floatCount = 640 * 480 * 3;
  float* data = reinterpret_cast<float*>(image.GetData());
  float max = 0;

  for (size_t i = 0; i < floatCount; ++i)
  {
    if (data[i] > max) max = data[i];
  }

  for (size_t i = 0; i < floatCount; ++i)
  {
    data[i] /= max;
  }

  image.Save("image_10_scaled_albedos_with_final_lighting.png");

  camera->CaptureLighting(image);
  image.Save("image_11_final_lighting.png");

  data = reinterpret_cast<float*>(image.GetData());

  for (size_t i = 0; i < floatCount; ++i)
  {
    if (data[i] > max) max = data[i];
  }

  for (size_t i = 0; i < floatCount; ++i)
  {
    data[i] /= max;
  }

  image.Save("image_12_scaled_albedos_with_final_lighting.png");

  camera->Capture(image);
  image.Save("image_13_render_with_final_lighting_and_final_albedos.png");

  // std::cout << std::endl;
  // std::cout << std::endl;

  // for (size_t i = 0; i < light->GetVoxelCount(); ++i)
  // {
  //   const Vector rgb = rvalues[i].GetRGB();
  //   std::cout << rgb[0] << " ";
  //   std::cout << rgb[1] << " ";
  //   std::cout << rgb[2] << std::endl;
  // }

  // std::cout << std::endl;
  // std::cout << std::endl;

  // for (size_t i = 0; i < light->GetVoxelCount(); ++i)
  // {
  //   std::cout << "Voxel Value " << i << ": ";
  //   std::cout << lynx::Get(values + 3 * i + 0) << " ";
  //   std::cout << lynx::Get(values + 3 * i + 1) << " ";
  //   std::cout << lynx::Get(values + 3 * i + 2) << std::endl;
  // }

  // ASSERT_TRUE(summary.solutionUsable && summary.finalCost < 1E-6);
}

} // namespace testing

} // namespace torch