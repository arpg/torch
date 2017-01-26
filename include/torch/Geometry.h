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

    void PreBuildScene() override;

    void BuildScene(Link& link) override;

    void PostBuildScene() override;

    void UpdateTransform() override;
};

} // namespace torch