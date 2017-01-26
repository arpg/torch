#include <iostream>
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

  std::shared_ptr<PointLight> light;
  light = scene.CreatePointLight();
  light->SetIntensity(20, 20, 20);
  light->SetPosition(4, -2, -1);
  scene.Add(light);

  std::shared_ptr<Sphere> geometry;
  geometry = scene.CreateSphere();
  geometry->SetOrientation(0, 0, 0);
  geometry->SetPosition(0, 0, 0);
  geometry->SetScale(5, 3, 5);

  std::shared_ptr<MatteMaterial> material;
  material = scene.CreateMatteMaterial();
  material->SetAlbedo(1, 0, 0);

  std::shared_ptr<Primitive> primitive;
  primitive = scene.CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  primitive->SetOrientation(0, 0, 0);
  primitive->SetPosition(0, 0, 5);
  primitive->SetScale(1.5, 1, 1);
  group->AddChild(primitive);

  std::shared_ptr<Camera> camera;
  camera = scene.CreateCamera();
  camera->SetImageSize(640, 480);
  camera->SetFocalLength(320, 320);
  camera->SetCenterPoint(320, 240);
  camera->SetOrientation(0, 0, 0);
  camera->SetPosition(0, 0, 0);
  camera->SetSampleCount(128);

  Image image;
  camera->Capture(image);
  image.Save("image.png");

  std::cout << "Finished." << std::endl;
  return 0;
}