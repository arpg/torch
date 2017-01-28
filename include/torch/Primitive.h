#pragma once

#include <torch/Node.h>

namespace torch
{

class Primitive : public Node
{
  public:

    Primitive(std::shared_ptr<Context> context);

    std::shared_ptr<Geometry> GetGeometry() const;

    void SetGeometry(std::shared_ptr<Geometry> geometry);

    std::shared_ptr<Material> GetMaterial() const;

    void SetMaterial(std::shared_ptr<Material> material);

    BoundingBox GetBounds(const Transform &transform) override;

    void BuildScene(Link& link) override;

  protected:

    void ValidateChildren();

  protected:

    std::shared_ptr<Geometry> m_geometry;

    std::shared_ptr<Material> m_material;
};

} // namespace torch