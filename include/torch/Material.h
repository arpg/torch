#pragma once

#include <string>
#include <torch/Object.h>

namespace torch
{

class Material : public Object
{
  public:

    Material(std::shared_ptr<Context> context, const std::string& name);

    ~Material();

    std::string GetName() const;

    void PreBuildScene() override;

    void BuildScene(Link& link) override;

    void PostBuildScene() override;

  private:

    void Initialize();

  protected:

    const std::string m_name;

    optix::Material m_material;
};

} // namespace torch