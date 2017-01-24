#pragma once

#include <memory>

namespace torch
{

class Context;

class Object
{
  public:

    Object(std::shared_ptr<Context> context);

    ~Object();

  protected:

    std::shared_ptr<Context> m_context;
};

} // namespace torch