#pragma once

#include <vector>
#include <optixu/optixpp.h>
#include <torch/Link.h>
#include <torch/Object.h>
#include <torch/Transformable.h>

namespace torch
{

class Node : public Object, public Transformable
{
  public:

    Node(std::shared_ptr<Context> context);

    ~Node();

    size_t GetChildCount() const;

    std::shared_ptr<Node> GetChild(size_t index) const;

    bool HasChild(std::shared_ptr<const Node> child) const;

    void AddChild(std::shared_ptr<Node> child);

    void RemoveChild(std::shared_ptr<const Node> child);

    void RemoveChildren();

    virtual void BuildScene(optix::Variable variable);

    void PreBuildScene() override;

    void BuildScene(Link& link) override;

    void PostBuildScene() override;

  protected:

    void BuildChildScene(Link& link);

    bool CanAddChild(std::shared_ptr<const Node> child) const;

    void UpdateTransform() override;

  protected:

    std::vector<std::shared_ptr<Node>> m_children;
};

} // namespace torch