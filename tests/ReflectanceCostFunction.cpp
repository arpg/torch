#include <gtest/gtest.h>
#include <lynx/lynx.h>
#include <torch/Torch.h>

namespace torch
{

namespace testing
{

TEST(ReflectanceCostFunction, Gradient)
{
  std::vector<Point> vertices;
  vertices.push_back(Point( 0,  0,  0));
  vertices.push_back(Point( 1, -1,  0));
  vertices.push_back(Point( 1,  1,  0));
  vertices.push_back(Point(-1,  1,  0));
  vertices.push_back(Point(-1, -1,  0));

  std::vector<Normal> normals;
  normals.push_back(Normal( 0,  0, -1));
  normals.push_back(Normal( 0,  0, -1));
  normals.push_back(Normal( 0,  0, -1));
  normals.push_back(Normal( 0,  0, -1));
  normals.push_back(Normal( 0,  0, -1));

  std::vector<uint3> faces;
  faces.push_back(make_uint3(0, 1, 2));
  faces.push_back(make_uint3(0, 2, 3));
  faces.push_back(make_uint3(0, 3, 4));
  faces.push_back(make_uint3(0, 4, 1));

  std::vector<Spectrum> albedos;
  albedos.push_back(Spectrum::FromRGB(0.55, 0.55, 0.55));
  albedos.push_back(Spectrum::FromRGB(0.54, 0.54, 0.54));
  albedos.push_back(Spectrum::FromRGB(0.53, 0.53, 0.53));
  albedos.push_back(Spectrum::FromRGB(0.52, 0.52, 0.52));
  albedos.push_back(Spectrum::FromRGB(0.51, 0.51, 0.51));

  Scene scene;

  std::shared_ptr<Mesh> geometry;
  geometry = scene.CreateMesh();
  geometry->SetVertices(vertices);
  geometry->SetNormals(normals);
  geometry->SetFaces(faces);

  std::shared_ptr<MatteMaterial> material;
  material = scene.CreateMatteMaterial();
  material->SetAlbedos(albedos);

  std::shared_ptr<Primitive> primitive;
  primitive = scene.CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  scene.Add(primitive);

  float* values;
  const size_t bytes = sizeof(Spectrum) * albedos.size();
  LYNX_CHECK_CUDA(cudaMalloc(&values, bytes));
  LYNX_CHECK_CUDA(cudaMemcpy(values, albedos.data(), bytes,
      cudaMemcpyHostToDevice));

  lynx::Problem problem;

  problem.AddParameterBlock(values, 3 * albedos.size());
  problem.SetLowerBound(values, 0.0f);
  problem.SetUpperBound(values, 1.0f);

  ReflectanceCostFunction* costFunction;
  costFunction = new ReflectanceCostFunction(material, geometry);
  costFunction->SetChromaticityThreshold(0.9f);
  costFunction->SetWeight(1.0f);
  problem.AddResidualBlock(costFunction, nullptr, values);

  problem.CheckGradients();

  LYNX_CHECK_CUDA(cudaFree(values));
}

TEST(ReflectanceCostFunction, Optimization)
{
  std::vector<Point> vertices;
  vertices.push_back(Point( 0,  0,  0));
  vertices.push_back(Point( 1, -1,  0));
  vertices.push_back(Point( 1,  1,  0));
  vertices.push_back(Point(-1,  1,  0));
  vertices.push_back(Point(-1, -1,  0));

  std::vector<Normal> normals;
  normals.push_back(Normal( 0,  0, -1));
  normals.push_back(Normal( 0,  0, -1));
  normals.push_back(Normal( 0,  0, -1));
  normals.push_back(Normal( 0,  0, -1));
  normals.push_back(Normal( 0,  0, -1));

  std::vector<uint3> faces;
  faces.push_back(make_uint3(0, 1, 2));
  faces.push_back(make_uint3(0, 2, 3));
  faces.push_back(make_uint3(0, 3, 4));
  faces.push_back(make_uint3(0, 4, 1));

  std::vector<Spectrum> albedos;
  albedos.push_back(Spectrum::FromRGB(0.55, 0.55, 0.55));
  albedos.push_back(Spectrum::FromRGB(0.54, 0.54, 0.54));
  albedos.push_back(Spectrum::FromRGB(0.53, 0.53, 0.53));
  albedos.push_back(Spectrum::FromRGB(0.52, 0.52, 0.52));
  albedos.push_back(Spectrum::FromRGB(0.51, 0.51, 0.51));

  Scene scene;

  std::shared_ptr<Mesh> geometry;
  geometry = scene.CreateMesh();
  geometry->SetVertices(vertices);
  geometry->SetNormals(normals);
  geometry->SetFaces(faces);

  std::shared_ptr<MatteMaterial> material;
  material = scene.CreateMatteMaterial();
  material->SetAlbedos(albedos);

  std::shared_ptr<Primitive> primitive;
  primitive = scene.CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  scene.Add(primitive);

  float* values;
  const size_t bytes = sizeof(Spectrum) * albedos.size();
  LYNX_CHECK_CUDA(cudaMalloc(&values, bytes));
  LYNX_CHECK_CUDA(cudaMemcpy(values, albedos.data(), bytes,
      cudaMemcpyHostToDevice));

  lynx::Problem problem;

  problem.AddParameterBlock(values, 3 * albedos.size());
  problem.SetLowerBound(values, 0.0f);
  problem.SetUpperBound(values, 1.0f);

  ReflectanceCostFunction* costFunction;
  costFunction = new ReflectanceCostFunction(material, geometry);
  costFunction->SetChromaticityThreshold(0.9f);
  costFunction->SetWeight(1.0f);
  problem.AddResidualBlock(costFunction, nullptr, values);

  lynx::Solver::Summary summary;
  lynx::Solver solver(&problem);
  solver.Solve(&summary);

  ASSERT_TRUE(summary.solutionUsable && summary.finalCost < 1E-6);

  LYNX_CHECK_CUDA(cudaFree(values));
}

} // namespace testing

} // namespace torch