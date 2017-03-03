#include <string>
#include <regex>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <torch/Exception.h>
#include <optix_math.h>

DEFINE_bool(show, false, "show processed images");
DEFINE_string(out_dir, ".", "output directory for processed images");
DEFINE_string(out_prefix, "new_", "prefix to prepend to processed images");

void GetFileParts(const std::string& file, std::string& dir, std::string& name)
{
  std::smatch match;
  std::regex expression("^(.*/)([^/]+)$");
  TORCH_ASSERT(std::regex_search(file, match, expression), "invalid file name");
  dir = match[1].str();
  name = match[2].str();
}

std::string GetOuputFileName(const std::string& file)
{
  std::string dir, name;
  GetFileParts(file, dir, name);
  return FLAGS_out_dir + "/" + FLAGS_out_prefix + name;
}

float GetPolynomial(float x, float a, float b, float c)
{
  const float x2 = x * x;
  const float x3 = x2 * x;
  return a * x3 + b * x2 + c * x;
}

void ConvertToIrradiance(float3& pixel)
{
  pixel.x = GetPolynomial(pixel.x, 0.5154f, -0.0500f, 0.5346f);
  pixel.y = GetPolynomial(pixel.y, 0.4746f, -0.0074f, 0.5329f);
  pixel.z = GetPolynomial(pixel.z, 0.7385f, -0.3758f, 0.6373f);
}

void RemoveVignetting(float3& pixel, int x, int y, int w, int h)
{
  const float w2 = w / 2.0f;
  const float h2 = h / 2.0f;
  const float rmax = sqrt(w2 * w2 + h2 * h2);
  const float u = (x + 0.5f - w2) / rmax;
  const float v = (y + 0.5f - h2) / rmax;
  const float r = sqrtf(u * u + v * v);
  const float s = 1 + GetPolynomial(r, -0.8172, 2.0096, -0.1249);
  pixel.x *= s;
  pixel.y *= s;
  pixel.z *= s;
}

void ProcessImage(const std::string& file)
{
  const std::string out_file = GetOuputFileName(file);
  LOG(INFO) << "Writing to " << out_file << "..." << std::endl;

  cv::Mat image = cv::imread(file);
  image.convertTo(image, CV_32FC3, 1.0f / 255);
  float3* data = reinterpret_cast<float3*>(image.data);

  const int w = image.cols;
  const int h = image.rows;

  #pragma omp parallel for
  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      const size_t index = y * w + x;
      float3& pixel = data[index];
      ConvertToIrradiance(pixel);
      RemoveVignetting(pixel, x, y, w, h);
    }
  }

  if (FLAGS_show)
  {
    cv::imshow("Image", image);
    int key = cv::waitKey(1);
    if (key == 27) std::exit(0);
  }

  image.convertTo(image, CV_8UC3, 255);
  TORCH_ASSERT(cv::imwrite(out_file, image), "unabled to write image");
}

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const int image_count = argc - 1;
  LOG(INFO) << "Processing " << image_count << " images..." << std::endl;

  for (int i = 0; i < image_count; ++i)
  {
    ProcessImage(argv[i + 1]);
  }

  LOG(INFO) << "Success" << std::endl;
  return 0;
}