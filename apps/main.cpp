#include <iostream>
#include <vector>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  std::cout << "Starting..." << std::endl;

  Scene scene;
  scene.SetEpsilon(1E-4);

  const Spectrum sky = Spectrum::FromRGB(0.1, 0.1, 0.5);
  const Spectrum sun = Spectrum::FromRGB(1.0, 1.0, 0.8);
  std::shared_ptr<EnvironmentLight> envLight;
  envLight = scene.CreateEnvironmentLight();
  envLight->SetRowCount(21);
  envLight->SetRadiance(sky);
  envLight->SetRadiance(150, sun);
  scene.Add(envLight);

  std::shared_ptr<Mesh> mesh;
  mesh = scene.CreateMesh("bunny.ply");
  mesh->SetOrientation(0, 0.8 * M_PIf, M_PIf);
  mesh->SetPosition(-0.75, 1.5, 3.5);
  mesh->SetScale(10);

  std::shared_ptr<MatteMaterial> material3;
  material3 = scene.CreateMatteMaterial();
  material3->SetAlbedo(1.0, 1.0, 1.0);

  std::shared_ptr<Primitive> primitive3;
  primitive3 = scene.CreatePrimitive();
  primitive3->SetGeometry(mesh);
  primitive3->SetMaterial(material3);
  primitive3->SetOrientation(0, 0, 0);
  primitive3->SetPosition(0, 0, 0);
  scene.Add(primitive3);

  std::shared_ptr<Group> group;
  group = scene.CreateGroup();
  group->SetOrientation(0, 0, 0);
  group->SetPosition(0, 0, 0);
  scene.Add(group);

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
