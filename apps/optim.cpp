#include <iostream>
#include <torch/Torch.h>

using namespace torch;

int main(int argc, char** argv)
{
  std::cout << "Starting..." << std::endl;

  Problem problem;
  problem.ComputeAlbedoDerivatives();

  std::cout << "Finished." << std::endl;
  return 0;
}