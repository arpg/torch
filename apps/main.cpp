#include <iostream>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  Scene scene;

  std::shared_ptr<Camera> camera;
  camera = scene.CreateCamera();
  camera->SetImageSize(640, 480);
  camera->SetFocalLength(320, 320);
  camera->SetCenterPoint(320, 240);
  camera->SetOrientation(0, 0, 0);
  camera->SetPosition(0, 0, 0);

  Image image;
  camera->Capture(image);
  image.Save("image.png");
  std::cout << "Success" << std::endl;

  return 0;
}