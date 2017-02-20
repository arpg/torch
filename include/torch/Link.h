#pragma once

#include <torch/Core.h>
#include <torch/Transform.h>

namespace torch
{

class Link
{
  public:

    Link(std::shared_ptr<Context> context);

    void Apply(optix::Material material);

    void Attach(optix::GeometryInstance instance);

    void Write(optix::Variable variable);

    Link Branch(const Transform& transform);

    Transform GetTransform() const;

    void Clear();

  private:

    void Initialize();

    void CreateGeometryGroup();

    void CreateGroup();

    optix::Acceleration CreateAcceleration();

  protected:

    std::shared_ptr<Context> m_context;

    Transform m_transform;

    optix::Material m_material;

    optix::Group m_group;

    optix::GeometryGroup m_geomGroup;
};

} // namespace torch