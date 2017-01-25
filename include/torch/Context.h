#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optixu/optixpp.h>

namespace torch
{

class Node;
class Object;

class Context : public std::enable_shared_from_this<Context>
{
  public:

    ~Context();

    void MarkDirty();

    void Launch(unsigned int id, RTsize w);

    void Launch(unsigned int id, RTsize w, RTsize h);

    void Launch(unsigned int id, RTsize w, RTsize h, RTsize d);

    void Launch(unsigned int id, const uint2& size);

    void Launch(unsigned int id, const uint3& size);

    static std::shared_ptr<Context> Create();

    optix::Acceleration CreateAcceleration();

    optix::Buffer CreateBuffer(unsigned int type);

    optix::Group CreateGroup();

    optix::Geometry CreateGeometry();

    optix::GeometryGroup CreateGeometryGroup();

    optix::GeometryInstance CreateGeometryInstance();

    optix::Material CreateMaterial();

    optix::Transform CreateTransform();

    optix::Variable GetVariable(const std::string& name);

    optix::Program GetProgram(const std::string& file, const std::string& name);

    optix::Program CreateProgram(const std::string& file,
        const std::string& name);

    unsigned int RegisterLaunchProgram(optix::Program program);

    void RegisterObject(std::shared_ptr<Object> object);

    std::shared_ptr<Node> GetSceneRoot() const;

  protected:

    void PrepareLaunch();

    void DropOrphans();

    void PreBuildScene();

    void BuildScene();

    void PostBuildScene();

  private:

    Context();

    void Initialize();

    void CreateContext();

    void CreateSceneRoot();

  protected:

    bool m_dirty;

    optix::Context m_context;

    std::shared_ptr<Node> m_sceneRoot;

    std::vector<std::shared_ptr<Object>> m_objects;

    std::unordered_map<std::string, optix::Program> m_programs;
};

} // namespace torch