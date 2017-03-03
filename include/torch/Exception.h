#pragma once

#include <exception>
#include <string>

#define TORCH_ASSERT(condition, message) \
{ if (!(condition)) throw ::torch::Exception(message); }

namespace torch
{

class Exception : public std::exception
{
  public:

    Exception(const std::string& message) :
      m_message(message)
    {
    }

    const char* what() const throw() override
    {
      return m_message.c_str();
    }

  protected:

    const std::string m_message;
};

} // namespace torch