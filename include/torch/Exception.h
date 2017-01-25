#pragma once

#include <exception>
#include <string>

namespace torch
{

class Exception : public std::exception
{
  public:

    Exception(const std::string& message) :
      m_message(message)
    {
    }

    ~Exception()
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