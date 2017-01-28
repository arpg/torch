#include <torch/BoundingBox.h>
#include <torch/Point.h>
#include <torch/Vector.h>

namespace torch
{

BoundingBox::BoundingBox() :
  bmin(+infinity, +infinity, +infinity),
  bmax(-infinity, -infinity, -infinity)
{
}

void BoundingBox::Union(const Point& p)
{
  bmin.x = std::min(p.x, bmin.x);
  bmin.y = std::min(p.y, bmin.y);
  bmin.z = std::min(p.z, bmin.z);
  bmax.x = std::max(p.x, bmax.x);
  bmax.y = std::max(p.y, bmax.y);
  bmax.z = std::max(p.z, bmax.z);
}

void BoundingBox::Union(float x, float y, float z)
{
  Union(Point(x, y, z));
}

void BoundingBox::Union(const BoundingBox& b)
{
  bmin.x = std::min(b.bmin.x, bmin.x);
  bmin.y = std::min(b.bmin.y, bmin.y);
  bmin.z = std::min(b.bmin.z, bmin.z);
  bmax.x = std::max(b.bmax.x, bmax.x);
  bmax.y = std::max(b.bmax.y, bmax.y);
  bmax.z = std::max(b.bmax.z, bmax.z);
}

float BoundingBox::GetRadius() const
{
  const Point center = 0.5 * bmax + bmin;
  return (bmax - center).Length();
}

} // namespace torch