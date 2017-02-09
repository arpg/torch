#include <torch/Image.h>

namespace torch
{

Image::Image()
{
}

void Image::Resize(unsigned int w, unsigned int h)
{
  m_image = cv::Mat(h, w, CV_32FC3);
}

unsigned int Image::GetWidth() const
{
  return m_image.cols;
}

unsigned int Image::GetHeight() const
{
  return m_image.rows;
}

size_t Image::GetByteCount() const
{
  return m_image.total() * m_image.elemSize();
}

unsigned char* Image::GetData() const
{
  return m_image.data;
}

void Image::Load(const std::string& file)
{
  cv::Mat temp = cv::imread(file);
  cv::cvtColor(temp, temp, CV_RGB2BGR);
  temp.convertTo(m_image, CV_32FC3, 1 / 255.0);
}

void Image::Save(const std::string& file)
{
  cv::Mat temp;
  m_image.convertTo(temp, CV_8UC3, 255);
  cv::cvtColor(temp, temp, CV_RGB2BGR);
  cv::imwrite(file, temp);
}

} // namespace torch