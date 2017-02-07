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

  std::vector<std::shared_ptr<Camera>> cameras;
  scene.GetCameras(cameras);
  Image image;

  for (size_t i = 0; i < cameras.size(); ++i)
  {
    std::cout << "Rendering image " << i << "..." << std::endl;
    const std::string file = "image" + std::to_string(i) + ".png";
    cameras[i]->Capture(image);
    image.Save(file);
  }

  const clock_t stop = clock();
  const double time = double(stop - start) / CLOCKS_PER_SEC;
  std::cout << "Elapsed Time: " << time << std::endl;
  return 0;
}
