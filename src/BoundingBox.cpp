#include <torch/BoundingBox.h>
#include <torch/Point.h>

namespace torch
{

BoundingBox::BoundingBox()
{
}

void BoundingBox::Add(const Point& p)
{
  bmin.x = std::min(p.x, bmin.x);
  bmin.y = std::min(p.y, bmin.y);
  bmin.z = std::min(p.z, bmin.z);

  bmax.x = std::max(p.x, bmax.x);
  bmax.y = std::max(p.y, bmax.y);
  bmax.z = std::max(p.z, bmax.z);
}

void BoundingBox::Add(float x, float y, float z)
{
  Add(Point(x, y, z));
}

void BoundingBox::Add(const BoundingBox& b)
{
  Add(b.bmin);
  Add(b.bmax);
}

} // namespace torch