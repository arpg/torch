#pragma once

#include <torch/Core.h>

namespace torch
{

class ShadowRemover
{
  public:

    struct Options
    {
    };

  public:

    ShadowRemover(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<MatteMaterial> material);

    virtual ~ShadowRemover();

    void Configure(const Options& options);

    void Remove();

  protected:

    Options m_options;

    std::shared_ptr<Mesh> m_mesh;

    std::shared_ptr<MatteMaterial> m_material;
};

} // namespace torc