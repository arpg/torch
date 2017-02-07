#include <torch/Transform.h>
#include <torch/BoundingBox.h>
#include <torch/Normal.h>
#include <torch/Point.h>
#include <torch/Vector.h>

namespace torch
{

Transform::Transform() :
  m_matrix(optix::Matrix4x4::identity())
{
}

Transform::Transform(const optix::Matrix4x4& matrix) :
  m_matrix(matrix)
{
}

Transform::Transform(optix::Transform transform)
{
  transform->getMatrix(false, m_matrix.getData(), nullptr);
}

optix::Matrix4x4 Transform::GetRotationMatrix() const
{
  unsigned int count = 0;
  optix::Matrix4x4 R = m_matrix;
  R.setCol(3, make_float4(0, 0, 0, 1));
  optix::Matrix4x4 Rnext;
  float norm = 0.0f;

  do
  {
    Rnext = 0.5f * (R + R.transpose().inverse());
    norm = GetNorm(R, Rnext);
    R = Rnext;
  }
  while (++count < 100 && norm > 1E-4);

  return R;
}

Vector Transform::GetEulerAngles() const
{
  Vector result;
  const optix::Matrix4x4 R = GetRotationMatrix();
  const float c2 = sqrt(R[0] * R[0] + R[4] * R[4]);
  result.x = atan2(R[9], R[10]);

  if (result.x > 0)
  {
    result.x = result.x - M_PIf;
    result.y = atan2(-R[8], -c2);
  }
  else
  {
    result.y = atan2(-R[8], c2);
  }

  const float s1 = sin(result.x);
  const float c1 = cos(result.x);
  result.z = atan2(s1 * R[2] - c1 * R[1], c1 * R[5] - s1 * R[6]);
  return -result;
}

void Transform::SetRotation(const optix::Matrix3x3& rotation)
{
  const optix::Matrix4x4 S = GetScaleMatrix();
  optix::Matrix4x4 M = GetTranslationMatrix();

  float* dst = M.getData();
  const float* src = rotation.getData();
  std::copy(&src[0], &src[3], &dst[0]);
  std::copy(&src[3], &src[6], &dst[4]);
  std::copy(&src[6], &src[9], &dst[8]);

  m_matrix = M * S;
}

void Transform::SetRotation(float x, float y, float z)
{
  const optix::Matrix4x4 R =
      optix::Matrix4x4::rotate(-z, make_float3(0, 0, 1)) *
      optix::Matrix4x4::rotate(-y, make_float3(0, 1, 0)) *
      optix::Matrix4x4::rotate(-x, make_float3(1, 0, 0));

  const optix::Matrix4x4 T = GetTranslationMatrix();
  const optix::Matrix4x4 S = GetScaleMatrix();
  m_matrix = T * R * S;
}

void Transform::SetRotation(const Vector& rotation)
{
  SetRotation(rotation.x, rotation.y, rotation.z);
}

Vector Transform::GetTranslation() const
{
  const float4 col = m_matrix.getCol(3);
  return Vector(col.x, col.y, col.z);
}

optix::Matrix4x4 Transform::GetTranslationMatrix() const
{
  optix::Matrix4x4 T = optix::Matrix4x4::identity();
  T.setCol(3, m_matrix.getCol(3));
  return T;
}

void Transform::SetTranslation(float x, float y, float z)
{
  SetTranslation(Vector(x, y, z));
}

void Transform::SetTranslation(const Vector& translation)
{
  float4 col;
  col.x = translation.x;
  col.y = translation.y;
  col.z = translation.z;
  col.w = 1;

  m_matrix.setCol(3, col);
}

optix::Matrix4x4 Transform::GetScaleMatrix() const
{
  const optix::Matrix4x4 R = GetRotationMatrix();
  return R.transpose() * m_matrix;
}

Vector Transform::GetScale() const
{
  const optix::Matrix4x4 S = GetScaleMatrix();
  return Vector(S[0], S[5], S[10]);
}

void Transform::SetScale(float x, float y, float z)
{
  SetScale(Vector(x, y, z));
}

void Transform::SetScale(const Vector& scale)
{
  const float3 s = make_float3(scale.x, scale.y, scale.z);
  const optix::Matrix4x4 T = GetTranslationMatrix();
  const optix::Matrix4x4 R = GetRotationMatrix();
  const optix::Matrix4x4 S = optix::Matrix4x4::scale(s);
  m_matrix = T * R * S;
}

void Transform::SetScale(float scale)
{
  SetScale(scale, scale, scale);
}

optix::Matrix4x4 Transform::GetMatrix() const
{
  return m_matrix;
}

optix::Matrix3x4 Transform::GetMatrix3x4() const
{
  return optix::Matrix3x4(m_matrix.getData());
}

Transform Transform::Inverse() const
{
  Transform result;
  result.m_matrix = m_matrix.inverse();
  return result;
}

Transform Transform::operator*(const Transform& transform) const
{
  Transform result;
  result.m_matrix = m_matrix * transform.m_matrix;
  return result;
}

Normal Transform::operator*(const Normal& normal) const
{
  float4 n = make_float4(normal.x, normal.y, normal.z, 0);
  n = m_matrix.inverse() * n;
  return Normal(n.x, n.y, n.z);
}

Point Transform::operator*(const Point& point) const
{
  const float4 p = m_matrix * make_float4(point.x, point.y, point.z, 1);
  return Point(p.x, p.y, p.z);
}

Vector Transform::operator*(const Vector& vector) const
{
  const float4 v = m_matrix * make_float4(vector.x, vector.y, vector.z, 0);
  return Vector(v.x, v.y, v.z);
}

BoundingBox Transform::operator*(const BoundingBox& b) const
{
  BoundingBox result;
  result.Union((*this) * Point(b.bmin.x, b.bmin.y, b.bmin.z));
  result.Union((*this) * Point(b.bmin.x, b.bmin.y, b.bmax.z));
  result.Union((*this) * Point(b.bmin.x, b.bmax.y, b.bmin.z));
  result.Union((*this) * Point(b.bmin.x, b.bmax.y, b.bmax.z));
  result.Union((*this) * Point(b.bmax.x, b.bmin.y, b.bmin.z));
  result.Union((*this) * Point(b.bmax.x, b.bmin.y, b.bmax.z));
  result.Union((*this) * Point(b.bmax.x, b.bmax.y, b.bmin.z));
  result.Union((*this) * Point(b.bmax.x, b.bmax.y, b.bmax.z));
  return result;
}

void Transform::Write(optix::Transform transform) const
{
  transform->setMatrix(false, m_matrix.getData(), nullptr);
}

void Transform::Write(optix::Variable variable) const
{
  variable->setMatrix4x4fv(false, m_matrix.getData());
}

float Transform::GetNorm(const optix::Matrix4x4& a, const optix::Matrix4x4& b)
{
  float norm = 0;

  for (unsigned int i = 0; i < 3; ++i)
  {
    float n = fabsf(a[4 * 0 + i] - b[4 * 0 + i]) +
              fabsf(a[4 * 1 + i] - b[4 * 1 + i]) +
              fabsf(a[4 * 2 + i] - b[4 * 2 + i]);

    norm = fmaxf(norm, n);
  }

  return norm;
}

} // namespace torch