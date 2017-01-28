#pragma once

#include <torch/Core.h>

namespace torch
{

class MeshLoader
{
  public:

    MeshLoader(const std::string& file);

    void Load(std::shared_ptr<Mesh> mesh);

  protected:

    const std::string m_file;
};

} // namespace torch