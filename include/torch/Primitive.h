#pragma once

#include <torch/Node.h>

namespace torch
{

class Geometry;
class Material;

class Primitive : public Node
{
  public:

    Primitive(std::shared_ptr<Context> context);

    ~Primitive();

    std::shared_ptr<Geometry> GetGeometry() const;

    void SetGeometry(std::shared_ptr<Geometry> geometry);

    std::shared_ptr<Material> GetMaterial() const;

    void SetMaterial(std::shared_ptr<Material> material);

    void BuildScene(Link& link) override;

  protected:

    void Validate();

  protected:

    std::shared_ptr<Geometry> m_geometry;

    std::shared_ptr<Material> m_material;
};

} // namespace torch