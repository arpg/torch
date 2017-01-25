#pragma once

#include <vector>
#include <torch/Geometry.h>

namespace torch
{

class GeometryGroup : public Geometry
{
  public:

    GeometryGroup(std::shared_ptr<Context> context);

    ~GeometryGroup();

    size_t GetChildCount() const;

    std::shared_ptr<Geometry> GetChild(size_t index) const;

    bool HasChild(std::shared_ptr<const Geometry> child) const;

    void AddChild(std::shared_ptr<Geometry> child);

    void RemoveChild(std::shared_ptr<const Geometry> child);

    void RemoveChildren();

    void PreBuildScene() override;

    void BuildScene(Link& link) override;

    void PostBuildScene() override;

  protected:

    void BuildChildScene(Link& link);

    bool CanAddChild(std::shared_ptr<const Geometry> child) const;

  protected:

    std::vector<std::shared_ptr<Geometry>> m_children;
};

} // namespace torch