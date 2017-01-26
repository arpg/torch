#include <optix.h>

rtBuffer<unsigned char, 1> buffer;

RT_PROGRAM void HandleError()
{
  const unsigned int code = rtGetExceptionCode();

  if (code == RT_EXCEPTION_STACK_OVERFLOW)
  {
    const unsigned char message[] = "device stack overflow";
    memcpy(&buffer[0], message, sizeof(message));
  }
  else
  {
    rtPrintExceptionDetails();
  }
}