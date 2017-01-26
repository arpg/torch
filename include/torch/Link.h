#pragma once

#include <memory>
#include <optixu/optixpp.h>
#include <torch/Transform.h>

namespace torch
{

class Context;

class Link
{
  public:

    Link(std::shared_ptr<Context> context);

    void Apply(optix::Material material);

    void Attach(optix::Transform transform);

    void Attach(optix::GeometryInstance instance);

    void Write(optix::Transform transform) const;

    void Write(optix::Variable variable) const;

    Link Branch(const Transform& transform) const;

    Transform GetTransform() const;

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