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
    cameras[i]->CaptureMask(image);
    // cameras[i]->Capture(image);
    image.Save(file);

    // Image refImage;
    // refImage.Load("/home/mike/Desktop/research/reference.png");

    // Image difImage;
    // difImage.Load("/home/mike/Desktop/research/bounces.png");

    // float* maskData = reinterpret_cast<float*>(image.GetData());
    // float* refData = reinterpret_cast<float*>(refImage.GetData());
    // float* difData = reinterpret_cast<float*>(difImage.GetData());

    // const unsigned int elemCount = 3 * refImage.GetHeight() * refImage.GetWidth();

    // for (uint i = 0; i < elemCount; ++i)
    // {
    //   // refData[i] *= maskData[i];
    //   refData[i] = maskData[i] * (refData[i] - difData[i]);
    // }

    // refImage.Save("/home/mike/Desktop/masked_image.png");
  }

  const clock_t stop = clock();
  const double time = double(stop - start) / CLOCKS_PER_SEC;
  std::cout << "Elapsed Time: " << time << std::endl;
  return 0;
}
