#include <iostream>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  std::cout << "Building scene..." << std::endl;

  std::shared_ptr<Scene> scene;
  scene = std::make_shared<Scene>();

  const std::string mesh_file =
      "/home/mike/Documents/Blender/bunny_box_painted.ply";

  std::shared_ptr<Mesh> geometry;
  geometry = scene->CreateMesh(mesh_file);
  geometry->SetScale(1, 1, 1);

  std::shared_ptr<Material> material;
  material = scene->CreateMaterial(mesh_file);

  std::shared_ptr<MatteMaterial> matteMaterial;
  matteMaterial = std::static_pointer_cast<MatteMaterial>(material);

  std::shared_ptr<Primitive> primitive;
  primitive = scene->CreatePrimitive();
  primitive->SetGeometry(geometry);
  primitive->SetMaterial(material);
  scene->Add(primitive);

  std::shared_ptr<Sphere> sphere;
  sphere = scene->CreateSphere();
  sphere->SetScale(0.2);

  std::shared_ptr<AreaLight> light;
  light = scene->CreateAreaLight();
  // light->SetPosition(0, 0, 0);
  light->SetPosition(-0.1, -0.075, 0.35);
  light->SetRadiance(0.75, 0.75, 0.75);
  light->SetGeometry(sphere);
  scene->Add(light);

  std::shared_ptr<Camera> camera;
  camera = scene->CreateCamera();
  camera->SetOrientation(-0.15, 0.9 * M_PIf, M_PIf);
  camera->SetPosition(-0.1, -0.075, 0.35);
  camera->SetImageSize(640, 480);
  camera->SetFocalLength(450, 450);
  camera->SetCenterPoint(320, 240);
  camera->SetSampleCount(4);

  std::cout << "Rendering original albedos..." << std::endl;

  Image image;
  camera->CaptureAlbedo(image);
  // camera->CaptureNormals(image);
  camera->CaptureLighting(image);
  // camera->Capture(image);
  image.Save("original_albedos.png");

  // std::cout << "Baking albedos..." << std::endl;

  // AlbedoBaker baker(scene);
  // baker.SetSampleCount(16);
  // baker.Bake(matteMaterial, geometry);

  // std::cout << "Saving mesh..." << std::endl;

  // matteMaterial->LoadAlbedos();
  // MeshWriter writer(geometry, matteMaterial);
  // writer.Write("baked_mesh.ply");

  // std::cout << "Rendering results..." << std::endl;

  // Image image;
  // camera->CaptureAlbedo(image);
  // image.Save("result.png");

  std::cout << "Success" << std::endl;
  return 0;
}