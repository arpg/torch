#pragma once

#include <memory>

namespace torch
{

class Context;
class Camera;
class Group;
class Node;

class Scene
{
  public:

    Scene();

    ~Scene();

    void Add(std::shared_ptr<Node> node);

    std::shared_ptr<Camera> CreateCamera();

    std::shared_ptr<Group> CreateGroup();

  private:

    template <typename T>
    std::shared_ptr<T> CreateObject();

    void Initialize();

  protected:

    std::shared_ptr<Context> m_context;
};

} // namespace torch