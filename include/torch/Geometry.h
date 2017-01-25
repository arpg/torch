#pragma once

#include <string>
#include <torch/Object.h>
#include <torch/Transformable.h>

namespace torch
{

class Geometry : public Object, public Transformable
{
  public:

    Geometry(std::shared_ptr<Context> context);

    ~Geometry();

    void UpdateTransform() override;
};

class SingleGeometry : public Geometry
{
  public:

    SingleGeometry(std::shared_ptr<Context> context, const std::string& name);

    ~SingleGeometry();

    std::string GetName() const;

    void PreBuildScene() override;

    void BuildScene(Link& link) override;

    void PostBuildScene() override;

  private:

    void Initialize();

  protected:

    const std::string m_name;

    optix::Geometry m_geometry;
};

} // namespace torch