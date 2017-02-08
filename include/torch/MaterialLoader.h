#pragma once

#include <string>
#include <torch/Core.h>

namespace torch
{

class MaterialLoader
{
  public:

    MaterialLoader(std::shared_ptr<Context> context);

    std::shared_ptr<Material> Load(const std::string& file);

  protected:

    std::shared_ptr<Context> m_context;
};

} // namespace torch