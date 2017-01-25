#include <torch/Scene.h>
#include <torch/Camera.h>
#include <torch/Context.h>
#include <torch/GeometryGroup.h>
#include <torch/Group.h>
#include <torch/MatteMaterial.h>
#include <torch/Primitive.h>
#include <torch/Sphere.h>

namespace torch
{

Scene::Scene()
{
  Initialize();
}

Scene::~Scene()
{
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

std::shared_ptr<Primitive> Scene::CreatePrimitive()
{
  return CreateObject<Primitive>();
}

std::shared_ptr<GeometryGroup> Scene::CreateGeometryGroup()
{
  return CreateObject<GeometryGroup>();
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