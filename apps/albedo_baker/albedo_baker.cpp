#include <iostream>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  std::cout << "Building scene..." << std::endl;

  std::shared_ptr<Scene> scene;
  scene = std::make_shared<Scene>();


  std::shared_ptr<Mesh> geometry;
  geometry = scene->CreateMesh("../shark.ply");
  geometry->SetScale(1, 1, 1);

  std::shared_ptr<Material> material;
  material = scene->CreateMaterial("../shark.ply");

  std::shared_ptr<MatteMaterial> matteMaterial;
  matteMaterial = std::static_pointer_cast<MatteMaterial>(material);

  std::shared_ptr<Primitive> primitive;
  primitive = scene->CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  scene->Add(primitive);

  std::shared_ptr<EnvironmentLight> light;
  light = scene->CreateEnvironmentLight();
  light->SetRowCount(21);
  light->SetRadiance(0.001, 0.001, 0.001);
  light->SetRadiance(67 , Spectrum::FromRGB(4.0, 4.0, 4.0));
  scene->Add(light);

  std::shared_ptr<Camera> camera;
  camera = scene->CreateCamera();
  camera->SetOrientation(0, 0, 0);
  camera->SetPosition(0, 0, 0);
  camera->SetImageSize(640, 480);
  camera->SetFocalLength(320, 320);
  camera->SetCenterPoint(320, 240);
  camera->SetSampleCount(4);

  std::cout << "Baking albedos..." << std::endl;

  AlbedoBaker baker(scene);
  baker.SetSampleCount(4);
  baker.Bake(matteMaterial, geometry);

  std::cout << "Rendering results..." << std::endl;

  Image image;
  camera->CaptureAlbedo(image);
  image.Save("result.png");

  std::cout << "Success" << std::endl;
  return 0;
}