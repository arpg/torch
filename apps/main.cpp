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

  std::shared_ptr<PointLight> light1;
  light1 = scene.CreatePointLight();
  light1->SetIntensity(50, 50, 50);
  light1->SetPosition(4, -1, -1);
  scene.Add(light1);

  std::shared_ptr<PointLight> light2;
  light2 = scene.CreatePointLight();
  light2->SetIntensity(10, 10, 106);
  light2->SetPosition(-4, -2, -1);
  scene.Add(light2);

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
  primitive->SetOrientation(0, 0, 0);
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