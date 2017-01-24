#pragma once

#include <vector>
#include <torch/Object.h>

namespace torch
{

class Node : public Object
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

  protected:

    bool CanAddChild(std::shared_ptr<const Node> child) const;

  protected:

    std::vector<std::shared_ptr<Node>> m_children;
};

} // namespace torch