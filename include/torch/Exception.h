#pragma once

#include <exception>
#include <string>

#define TORCH_ASSERT(cond, message) \
{ if (!(cond)) throw ::torch::Exception(__FILE__, __LINE__, message); }

#define TORCH_THROW(message) \
{ throw ::torch::Exception(__FILE__, __LINE__, message); }

namespace torch
{

class Exception : public std::exception
{
  public:

    Exception(const std::string& message);

    Exception(const std::string& file, int line, const std::string& message);

    const char* what() const throw() override;

  private:

    static std::string CreateMessage(const std::string& file, int line,
        const std::string& message);

  protected:

    std::string m_message;
};

} // namespace torch