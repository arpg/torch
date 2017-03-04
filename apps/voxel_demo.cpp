#include <iostream>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  Scene scene;

  std::shared_ptr<VoxelLight> light;
  light = scene.CreateVoxelLight();
  light->SetDimensions(3, 3, 3);
  light->SetVoxelSize(1);
  light->SetRadiance(0.0, 0.0, 0.0);
  light->SetRadiance(1, Spectrum::FromRGB(50, 0, 300));
  light->SetRadiance(7, Spectrum::FromRGB(50, 300, 0));
  light->SetOrientation(0, 0, 0);
  light->SetPosition(0, 0, 0);
  scene.Add(light);

  std::shared_ptr<Sphere> geometry;
  geometry = scene.CreateSphere();
  geometry->SetScale(1, 1, 1);

  std::shared_ptr<MatteMaterial> material;
  material = scene.CreateMatteMaterial();
  material->SetAlbedo(0.5, 0.1, 0.1);

  std::shared_ptr<Primitive> primitive;
  primitive = scene.CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  primitive->SetOrientation(0, 0, 0);
  primitive->SetPosition(0, 0, 3);
  scene.Add(primitive);

  std::shared_ptr<Sphere> geometry2;
  geometry2 = scene.CreateSphere();
  geometry2->SetScale(4, 4, 4);

  std::shared_ptr<MatteMaterial> material2;
  material2 = scene.CreateMatteMaterial();
  material2->SetAlbedo(1, 1, 1);

  std::shared_ptr<Primitive> primitive2;
  primitive2 = scene.CreatePrimitive();
  primitive2->SetGeometry(geometry2);
  primitive2->SetMaterial(material2);
  primitive2->SetOrientation(0, 0, 0);
  primitive2->SetPosition(0, 0, 7);
  scene.Add(primitive2);

  std::shared_ptr<Camera> camera;
  camera = scene.CreateCamera();
  camera->SetImageSize(640, 480);
  camera->SetFocalLength(320, 320);
  camera->SetCenterPoint(320, 240);
  camera->SetOrientation(0, -M_PIf / 8, 0);
  camera->SetPosition(1, 0, 0);
  camera->SetSampleCount(6);
  scene.Add(camera);

  Image image;
  camera->Capture(image);
  image.Save("image.png");

  std::cout << "Success" << std::endl;
  return 0;
}