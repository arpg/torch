#include <torch/Scene.h>
#include <torch/Camera.h>
#include <torch/Context.h>

namespace torch
{

Scene::Scene()
{
  Initialize();
}

Scene::~Scene()
{
}

std::shared_ptr<Camera> Scene::CreateCamera()
{
  return std::make_shared<Camera>(m_context);
}

void Scene::Initialize()
{
  m_context = Context::Create();
}

} // namespace torch