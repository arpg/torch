#pragma once

#include <string>
#include <torch/Geometry.h>

namespace torch
{

class SingleGeometry : public Geometry
{
  public:

    SingleGeometry(std::shared_ptr<Context> context, const std::string& name);

    ~SingleGeometry();

    std::string GetName() const;

    void BuildScene(Link& link) override;

  private:

    void Initialize();

  protected:

    const std::string m_name;

    optix::Geometry m_geometry;
};

} // namespace torch