#pragma once

#include <memory>

namespace torch
{

class Context;
class Camera;

class Scene
{
  public:

    Scene();

    ~Scene();

    std::shared_ptr<Camera> CreateCamera();

  private:

    void Initialize();

  protected:

    std::shared_ptr<Context> m_context;
};

} // namespace torch