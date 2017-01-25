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

    ~Link();

    void Attach(optix::Group group);

    void Attach(optix::GeometryGroup group);

    void Attach(optix::GeometryInstance instance);

    void Attach(optix::Transform transform);

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

    optix::Group m_group;

    optix::GeometryGroup m_geomGroup;
};

} // namespace torch