#include <ctime>
#include <iostream>
#include <vector>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  const clock_t start = clock();
  std::cout << "Starting..." << std::endl;

  Scene scene;
  SceneLoader loader("scene.xml");
  loader.Load(scene);

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

  const clock_t stop = clock();
  const double time = double(stop - start) / CLOCKS_PER_SEC;
  std::cout << "Elapsed Time: " << time << std::endl;
  return 0;
}
