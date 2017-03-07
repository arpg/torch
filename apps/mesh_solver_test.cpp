#include <iostream>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  std::cout << "Starting..." << std::endl;

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
  primitive->SetPosition(0, 0, 1);
  scene.Add(primitive);

  std::shared_ptr<VoxelLight> light;
  light = scene.CreateVoxelLight();
  light->SetDimensions(2, 2, 2);
  light->SetVoxelSize(0.1);
  light->SetRadiance(1.00, 2.01, 0.62);
  scene.Add(light);

  light->GetContext()->Compile();
  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  float* values = reinterpret_cast<float*>(pointer);

  lynx::Problem problem;

  problem.AddParameterBlock(values, 3 * light->GetVoxelCount());
  problem.SetLowerBound(values, 0.0f);

  MeshCostFunction* costFunction;
  costFunction = new MeshCostFunction(light, geometry, material);
  costFunction->SetMaxNeighborCount(10);
  costFunction->SetMaxNeighborDistance(10.0f);
  costFunction->SetSimilarityThreshold(0.0f);
  costFunction->SetLightSampleCount(1000);
  problem.AddResidualBlock(costFunction, nullptr, values);

  problem.CheckGradients();

  std::cout << "Success" << std::endl;
}