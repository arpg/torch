#pragma once

#include <torch/Core.h>

namespace torch
{

class Scene
{
  public:

    Scene();

    float GetEpsilon() const;

    void SetEpsilon(float epsilon);

    void Add(std::shared_ptr<Node> node);

    std::shared_ptr<Camera> CreateCamera();

    std::shared_ptr<Group> CreateGroup();

    std::shared_ptr<AreaLight> CreateAreaLight();

    std::shared_ptr<DistantLight> CreateDistantLight();

    std::shared_ptr<EnvironmentLight> CreateEnvironmentLight();

    std::shared_ptr<PointLight> CreatePointLight();

    std::shared_ptr<Primitive> CreatePrimitive();

    std::shared_ptr<GeometryGroup> CreateGeometryGroup();

    std::shared_ptr<Mesh> CreateMesh();

    std::shared_ptr<Mesh> CreateMesh(const std::string& file);

    std::shared_ptr<Sphere> CreateSphere();

    std::shared_ptr<MatteMaterial> CreateMatteMaterial();

  private:

    template <typename T>
    std::shared_ptr<T> CreateObject();

    void Initialize();

  protected:

    std::shared_ptr<Context> m_context;
};

} // namespace torch