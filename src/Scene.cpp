#include <torch/Scene.h>
#include <torch/AreaLight.h>
#include <torch/Camera.h>
#include <torch/Context.h>
#include <torch/DirectionalLight.h>
#include <torch/EnvironmentLight.h>
#include <torch/GeometryGroup.h>
#include <torch/Group.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/MeshLoader.h>
#include <torch/PointLight.h>
#include <torch/Primitive.h>
#include <torch/Sphere.h>

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

std::shared_ptr<Camera> Scene::CreateCamera()
{
  return CreateObject<Camera>();
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

std::shared_ptr<Primitive> Scene::CreatePrimitive()
{
  return CreateObject<Primitive>();
}

std::shared_ptr<GeometryGroup> Scene::CreateGeometryGroup()
{
  return CreateObject<GeometryGroup>();
}

std::shared_ptr<Mesh> Scene::CreateMesh()
{
  return CreateObject<Mesh>();
}

std::shared_ptr<Mesh> Scene::CreateMesh(const std::__cxx11::string& file)
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

std::shared_ptr<MatteMaterial> Scene::CreateMatteMaterial()
{
  return CreateObject<MatteMaterial>();
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