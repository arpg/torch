#include <iostream>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  std::cout << "Starting..." << std::endl;

  std::shared_ptr<Scene> scene;
  scene = std::make_shared<Scene>();

  std::shared_ptr<Mesh> mesh;
  mesh = scene->CreateMesh("shark.ply");

  std::shared_ptr<Material> material;
  material = scene->CreateMaterial("shark.ply");

  std::shared_ptr<Primitive> primitive;
  primitive = scene->CreatePrimitive();
  primitive->SetGeometry(mesh);
  primitive->SetMaterial(material);
  scene->Add(primitive);

  std::shared_ptr<MatteMaterial> matteMaterial;
  matteMaterial = std::static_pointer_cast<MatteMaterial>(material);

  std::shared_ptr<EnvironmentLight> light;
  light = scene->CreateEnvironmentLight();
  light->SetRowCount(21);
  light->SetRadiance(1E-4, 1E-4, 1E-4);
  scene->Add(light);

  std::shared_ptr<Camera> camera;
  camera = scene->CreateCamera();
  camera->SetOrientation(2.49978, 2.69522, -2.78359);
  camera->SetPosition(-0.4752, -0.476978, 0.342274);
  camera->SetImageSize(640, 480);
  camera->SetFocalLength(535.7239, 536.2900);
  camera->SetCenterPoint(320.2685, 240.2924);
  camera->SetSampleCount(1);

  std::shared_ptr<Image> image;
  image = std::make_shared<Image>();
  image->Load("reference.png");

  std::vector<std::shared_ptr<ReferenceImage>> references;
  references.push_back(std::make_shared<ReferenceImage>(camera, image));

  Problem problem(scene, mesh, matteMaterial, light, references);
  // problem.ComputeAlbedoDerivatives();
  problem.ComputeLightDerivatives();
  std::cout << "Finished." << std::endl;
  return 0;
}