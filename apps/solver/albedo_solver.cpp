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
  // albedos.push_back(Spectrum::FromRGB(0.55, 0.55, 0.55));
  // albedos.push_back(Spectrum::FromRGB(0.54, 0.54, 0.54));
  // albedos.push_back(Spectrum::FromRGB(0.53, 0.53, 0.53));
  // albedos.push_back(Spectrum::FromRGB(0.52, 0.52, 0.52));
  // albedos.push_back(Spectrum::FromRGB(0.51, 0.51, 0.51));
  albedos.push_back(Spectrum::FromRGB(0.5, 0.0, 0.0));
  albedos.push_back(Spectrum::FromRGB(0.5, 0.0, 0.0));
  albedos.push_back(Spectrum::FromRGB(0.5, 0.0, 0.0));
  albedos.push_back(Spectrum::FromRGB(0.5, 0.0, 0.0));
  albedos.push_back(Spectrum::FromRGB(0.5, 0.0, 0.0));

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

  std::shared_ptr<EnvironmentLight> light;
  light = scene.CreateEnvironmentLight();
  light->SetRowCount(3);
  light->SetRadiance(1.0, 1.0, 1.0);
  scene.Add(light);

  std::shared_ptr<Camera> camera;
  camera = scene.CreateCamera();
  camera->SetOrientation(0, 0, 0);
  camera->SetPosition(0, 0, 0);
  camera->SetImageSize(8, 8);
  camera->SetFocalLength(4, 4);
  camera->SetCenterPoint(4, 4);
  camera->SetSampleCount(10);

  std::shared_ptr<Image> image;
  image = std::make_shared<Image>();
  camera->Capture(*image);
  image->Save("reference.png");

  std::shared_ptr<Keyframe> keyframe;
  keyframe = std::make_shared<Keyframe>(camera, image);

  // albedos[0] = Spectrum::FromRGB(0.65, 0.15, 0.15);
  // albedos[1] = Spectrum::FromRGB(0.14, 0.58, 0.14);
  // albedos[2] = Spectrum::FromRGB(0.23, 0.83, 0.52);
  // albedos[3] = Spectrum::FromRGB(0.52, 0.52, 0.33);
  // albedos[4] = Spectrum::FromRGB(0.81, 0.11, 0.91);
  albedos[0] = Spectrum::FromRGB(0.1, 0.0, 0.0);
  albedos[1] = Spectrum::FromRGB(0.1, 0.0, 0.0);
  albedos[2] = Spectrum::FromRGB(0.1, 0.0, 0.0);
  albedos[3] = Spectrum::FromRGB(0.1, 0.0, 0.0);
  albedos[4] = Spectrum::FromRGB(0.1, 0.0, 0.0);

  material->SetAlbedos(albedos);
  material->GetContext()->Compile();

  std::shared_ptr<Image> image2;
  image2 = std::make_shared<Image>();
  camera->Capture(*image2);
  image2->Save("modified.png");

  optix::Buffer buffer = material->GetAlbedoBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  float* values = reinterpret_cast<float*>(pointer);

  lynx::Problem problem;

  problem.AddParameterBlock(values, 3 * material->GetAlbedoCount());
  problem.SetLowerBound(values, 0.0f);
  problem.SetUpperBound(values, 1.0f);

  AlbedoCostFunction* costFunction;
  costFunction = new AlbedoCostFunction(material, geometry);
  costFunction->AddKeyframe(keyframe);
  problem.AddResidualBlock(costFunction, nullptr, values);

  problem.CheckGradients();

  std::cout << "Success" << std::endl;
  return 0;
}
