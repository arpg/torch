#include <torch/Scene.h>
#include <torch/AreaLight.h>
#include <torch/Camera.h>
#include <torch/Context.h>
#include <torch/DirectionalLight.h>
#include <torch/EnvironmentLight.h>
#include <torch/GeometryGroup.h>
#include <torch/Group.h>
#include <torch/MaterialLoader.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/MeshLoader.h>
#include <torch/PointLight.h>
#include <torch/Primitive.h>
#include <torch/Sphere.h>
#include <torch/VoxelLight.h>

namespace torch
{

Scene::Scene()
{
  Initialize();
}

void Scene::Add(std::shared_ptr<Node> node)
{
  std::shared_ptr<Node> sceneRoot;
  sceneRoot = m_context->GetSceneRoot();
  sceneRoot->AddChild(node);
}

void Scene::GetCameras(std::vector<std::shared_ptr<Camera>>& cameras)
{
  size_t index = 0;
  cameras.reserve(m_cameras.size());

  for (size_t i = 0; i < m_cameras.size(); ++i)
  {
    std::weak_ptr<Camera> camera = m_cameras[i];

    if (camera.use_count() > 0)
    {
      cameras.push_back(camera.lock());
      m_cameras[index++] = camera;
    }
  }

  m_cameras.resize(index);
}

std::shared_ptr<Camera> Scene::CreateCamera()
{
  std::shared_ptr<Camera> camera;
  camera = CreateObject<Camera>();
  m_cameras.push_back(camera);
  return camera;
}

std::shared_ptr<Group> Scene::CreateGroup()
{
  return CreateObject<Group>();
}

std::shared_ptr<AreaLight> Scene::CreateAreaLight()
{
  return CreateObject<AreaLight>();
}

std::shared_ptr<DirectionalLight> Scene::CreateDirectionalLight()
{
  return CreateObject<DirectionalLight>();
}
std::shared_ptr<EnvironmentLight> Scene::CreateEnvironmentLight()
{
  return CreateObject<EnvironmentLight>();
}

std::shared_ptr<PointLight> Scene::CreatePointLight()
{
  return CreateObject<PointLight>();
}

std::shared_ptr<VoxelLight> Scene::CreateVoxelLight()
{
  return CreateObject<VoxelLight>();
}

std::shared_ptr<Primitive> Scene::CreatePrimitive()
{
  return CreateObject<Primitive>();
}

std::shared_ptr<Primitive> Scene::CreatePrimitive(const std::string& file)
{
  std::shared_ptr<Primitive> primitive = CreatePrimitive();
  primitive->SetGeometry(CreateMesh(file));
  primitive->SetMaterial(CreateMaterial(file));
  return primitive;
}

std::shared_ptr<GeometryGroup> Scene::CreateGeometryGroup()
{
  return CreateObject<GeometryGroup>();
}

std::shared_ptr<Mesh> Scene::CreateMesh()
{
  return CreateObject<Mesh>();
}

std::shared_ptr<Mesh> Scene::CreateMesh(const std::string& file)
{
  std::shared_ptr<Mesh> mesh = CreateMesh();
  MeshLoader loader(mesh);
  loader.Load(file);
  return mesh;
}

std::shared_ptr<Sphere> Scene::CreateSphere()
{
  return CreateObject<Sphere>();
}

std::shared_ptr<Material> Scene::CreateMaterial(const std::string& file)
{
  MaterialLoader loader(m_context);
  return loader.Load(file);
}

std::shared_ptr<MatteMaterial> Scene::CreateMatteMaterial()
{
  return CreateObject<MatteMaterial>();
}

optix::Context Scene::GetOptixContext()
{
  return m_context->GetContext();
}

std::shared_ptr<Context> Scene::GetContext()
{
  return m_context;
}

void Scene::Initialize()
{
  m_context = Context::Create();
}

template <typename T>
std::shared_ptr<T> Scene::CreateObject()
{
  std::shared_ptr<T> object;
  object = std::make_shared<T>(m_context);
  m_context->RegisterObject(object);
  return object;
}

} // namespace torch