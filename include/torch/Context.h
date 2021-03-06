#pragma once

#include <torch/Core.h>
#include <torch/BoundingBox.h>

namespace torch
{

class Context : public std::enable_shared_from_this<Context>
{
  public:

    void MarkDirty();

    void Compile();

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

    optix::Program GetProgram(const std::string& file, const std::string& name);

    optix::Program CreateProgram(const std::string& file,
        const std::string& name);

    optix::Variable GetVariable(const std::string& name);

    unsigned int RegisterLaunchProgram(optix::Program program);

    void RegisterObject(std::shared_ptr<Object> object);

    std::shared_ptr<Node> GetSceneRoot() const;

    std::shared_ptr<SceneLightSampler> GetLightSampler() const;

    std::shared_ptr<SceneGeometrySampler> GetGeometrySampler() const;

    BoundingBox GetSceneBounds() const;

    float GetSceneRadius() const;

    optix::Context GetContext();

  protected:

    void PrepareLaunch();

    void DropOrphans();

    void PreBuildScene();

    void BuildScene();

    void PostBuildScene();

    void FinishLaunch();

  private:

    Context();

    void Initialize();

    void CreateContext();

    void CreateSceneRoot();

    void CreateLightSampler();

    void CreateGeometrySampler();

    void CreateEmptyProgram();

    void CreateLink();

  protected:

    bool m_dirty;

    optix::Context m_context;

    std::shared_ptr<Node> m_sceneRoot;

    std::shared_ptr<SceneLightSampler> m_lightSampler;

    std::shared_ptr<SceneGeometrySampler> m_geometrySampler;

    std::vector<std::shared_ptr<Object>> m_objects;

    std::unordered_map<std::string, optix::Program> m_programs;

    optix::Program m_emptyProgram;

    unsigned int m_emptyProgramId;

    optix::Program m_errorProgram;

    optix::Buffer m_errorBuffer;

    BoundingBox m_bounds;

    std::unique_ptr<Link> m_link;
};

} // namespace torch