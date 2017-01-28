#include <iostream>
#include <vector>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  std::cout << "Starting..." << std::endl;

  Scene scene;
  scene.SetEpsilon(1E-4);

  std::shared_ptr<Group> group;
  group = scene.CreateGroup();
  group->SetOrientation(0, 0, 0);
  group->SetPosition(0, 0, 0);
  scene.Add(group);

  std::shared_ptr<DistantLight> distLight;
  distLight = scene.CreateDistantLight();
  distLight->SetDirection(-0.2, 0.1, 0.4);
  distLight->SetRadiance(2, 2, 2);
  scene.Add(distLight);

  // std::shared_ptr<DistantLight> distLight2;
  // distLight2 = scene.CreateDistantLight();
  // // distLight2->SetDirection(0.3, 0.8, 0.7);
  // distLight2->SetDirection(0, 0.5, 0.5);
  // distLight2->SetRadiance(2, 2, 2);
  // scene.Add(distLight2);

  // std::shared_ptr<Sphere> lightGeom;
  // lightGeom = scene.CreateSphere();
  // lightGeom->SetOrientation(0, 0, 0);
  // lightGeom->SetPosition(0, 0, 0);
  // lightGeom->SetScale(1);

  // std::shared_ptr<AreaLight> areaLight;
  // areaLight = scene.CreateAreaLight();
  // areaLight->SetGeometry(lightGeom);
  // areaLight->SetRadiance(0.5, 0.5, 0.5);
  // areaLight->SetPosition(4, -1, -1);
  // scene.Add(areaLight);

  // std::shared_ptr<PointLight> light1;
  // light1 = scene.CreatePointLight();
  // light1->SetIntensity(50, 50, 50);
  // light1->SetPosition(4, -1, -1);
  // scene.Add(light1);

  // std::shared_ptr<PointLight> light2;
  // light2 = scene.CreatePointLight();
  // light2->SetIntensity(10, 10, 106);
  // light2->SetPosition(-4, -2, -1);
  // scene.Add(light2);

  std::vector<float3> vertices;
  vertices.push_back(make_float3(0, 0, 0));
  vertices.push_back(make_float3(1, 1, 0));
  vertices.push_back(make_float3(1, 0, 0.2));
  vertices.push_back(make_float3(0, 1, 0));

  std::vector<float3> normals;
  normals.push_back(make_float3(0, 0, -1));
  normals.push_back(make_float3(0, 0, -1));
  normals.push_back(normalize(make_float3(0, -0.8, -0.2)));
  normals.push_back(make_float3(0, 0, -1));

  std::vector<uint3> faces;
  faces.push_back(make_uint3(0, 1, 2));
  faces.push_back(make_uint3(0, 3, 1));

  std::shared_ptr<Mesh> mesh;
  mesh = scene.CreateMesh();
  mesh->SetOrientation(0, 0, 0);
  mesh->SetPosition(-3, 0, 4);
  mesh->SetScale(1, 1, 1);
  mesh->SetVertices(vertices);
  mesh->SetNormals(normals);
  mesh->SetFaces(faces);

  std::shared_ptr<MatteMaterial> material3;
  material3 = scene.CreateMatteMaterial();
  material3->SetAlbedo(0.1, 0.1, 1.0);

  std::shared_ptr<Primitive> primitive3;
  primitive3 = scene.CreatePrimitive();
  primitive3->SetGeometry(mesh);
  primitive3->SetMaterial(material3);
  primitive3->SetOrientation(0, 0, 0);
  primitive3->SetPosition(0, 0, 0);
  scene.Add(primitive3);

  std::shared_ptr<Sphere> geometry;
  geometry = scene.CreateSphere();
  geometry->SetOrientation(0, 0, 0);
  geometry->SetPosition(0, 0, 0);
  geometry->SetScale(4, 2, 3);

  std::shared_ptr<MatteMaterial> material;
  material = scene.CreateMatteMaterial();
  material->SetAlbedo(1.0, 0.1, 0.1);

  std::shared_ptr<Primitive> primitive;
  primitive = scene.CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  primitive->SetOrientation(0, 0, M_PIf / 8);
  primitive->SetPosition(0, 0, 5);
  group->AddChild(primitive);

  std::shared_ptr<Sphere> geometry2;
  geometry2 = scene.CreateSphere();
  geometry2->SetOrientation(0, 0, 0);
  geometry2->SetPosition(0, 0, 0);
  geometry2->SetScale(1);

  std::shared_ptr<MatteMaterial> material2;
  material2 = scene.CreateMatteMaterial();
  material2->SetAlbedo(0.1, 0.5, 0.1);

  std::shared_ptr<Primitive> primitive2;
  primitive2 = scene.CreatePrimitive();
  primitive2->SetGeometry(geometry2);
  primitive2->SetMaterial(material2);
  primitive2->SetOrientation(0, 0, 0);
  primitive2->SetPosition(1.5, -0.5, 3.5);
  group->AddChild(primitive2);

  std::shared_ptr<Camera> camera;
  camera = scene.CreateCamera();
  camera->SetImageSize(640, 480);
  camera->SetFocalLength(320, 320);
  camera->SetCenterPoint(320, 240);
  camera->SetOrientation(0, 0, 0);
  camera->SetPosition(0, 0, 0);
  camera->SetSampleCount(512);

  Image image;
  camera->Capture(image);
  image.Save("image.png");
  std::cout << "Finished." << std::endl;
  return 0;
}