#pragma once

#include <string>
#include <memory>
#include <optixu/optixpp.h>

namespace torch
{

class Context
{
  public:

    ~Context();

    void MarkDirty();

    void Launch(unsigned int id, RTsize w);

    void Launch(unsigned int id, RTsize w, RTsize h);

    void Launch(unsigned int id, RTsize w, RTsize h, RTsize d);

    void Launch(unsigned int id, const uint2& size);

    void Launch(unsigned int id, const uint3& size);

    static std::shared_ptr<Context> Create();

    optix::Buffer CreateBuffer(unsigned int type);

    optix::Program CreateProgram(const std::string& file,
        const std::string& name);

    unsigned int RegisterLaunchProgram(optix::Program program);

  protected:

    void PrepareLaunch();

  private:

    Context();

    void Initialize();

    void CreateContext();

  protected:

    bool m_dirty;

    optix::Context m_context;
};

} // namespace torch