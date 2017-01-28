#pragma once

#include <torch/Object.h>
#include <torch/Transformable.h>

namespace torch
{

class Node : public Object, public Transformable
{
  public:

    Node(std::shared_ptr<Context> context);

    size_t GetChildCount() const;

    std::shared_ptr<Node> GetChild(size_t index) const;

    bool HasChild(std::shared_ptr<const Node> child) const;

    void AddChild(std::shared_ptr<Node> child);

    void RemoveChild(std::shared_ptr<const Node> child);

    void RemoveChildren();

    void PreBuildScene() override;

    void BuildScene(Link& link) override;

    void PostBuildScene() override;

    virtual void BuildScene(optix::Variable variable);

    virtual void GetBounds(BoundingBox& bounds);

  protected:

    virtual void GetBounds(const Transform& transform, BoundingBox& bounds);

    void BuildChildScene(Link& link);

    bool CanAddChild(std::shared_ptr<const Node> child) const;

    void UpdateTransform() override;

  protected:

    std::vector<std::shared_ptr<Node>> m_children;
};

} // namespace torch