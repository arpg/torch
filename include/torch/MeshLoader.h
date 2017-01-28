#pragma once

#include <torch/Core.h>

namespace torch
{

class MeshLoader
{
  public:

    MeshLoader(std::shared_ptr<Mesh> mesh);

    void Load(const std::string& file);

  protected:

    const std::shared_ptr<Mesh> m_mesh;
};

} // namespace torch