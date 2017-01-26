#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace torch
{

class Image
{
  public:

    Image();

    void Resize(unsigned int w, unsigned int h);

    unsigned int GetWidth() const;

    unsigned int GetHeight() const;

    size_t GetByteCount() const;

    unsigned char* GetData() const;

    void Load(const std::string& file);

    void Save(const std::string& file);

  protected:

    cv::Mat m_image;
};

} // namespace torch