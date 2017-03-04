#include <torch/Exception.h>

namespace torch
{

Exception::Exception(const std::string& message) :
  m_message(message)
{
}

Exception::Exception(const std::string& file, int line,
    const std::string& message) :
  m_message(CreateMessage(file, line, message))
{
}

const char* Exception::what() const throw()
{
  return m_message.c_str();
}

std::string Exception::CreateMessage(const std::string& file, int line,
    const std::string& message)
{
  return "In " + file + " on line " + std::to_string(line) + ": " + message;
}

} // namespace torch