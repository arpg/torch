#include <torch/Transform.h>

namespace torch
{

const unsigned int decompIters = 100;
const float decompEpsilon = 1E-4;

Transform::Transform() :
  m_matrix(optix::Matrix4x4::identity())
{
}

Transform::~Transform()
{
}

optix::Matrix4x4 Transform::GetRotationMatrix() const
{
  unsigned int count = 0;
  optix::Matrix4x4 R = m_matrix;
  optix::Matrix4x4 Rnext;
  float norm = 0.0f;

  do
  {
    Rnext = 0.5f * (R + R.transpose().inverse());
    norm = GetNorm(R, Rnext);
    R = Rnext;
  }
  while (++count < decompIters && norm > decompEpsilon);

  return R;
}

float3 Transform::GetEulerAngles() const
{
  float3 result;
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

void Transform::SetRotation(const float3& rotation)
{
  SetRotation(rotation.x, rotation.y, rotation.z);
}

float3 Transform::GetTranslation() const
{
  return make_float3(m_matrix.getCol(3));
}

optix::Matrix4x4 Transform::GetTranslationMatrix() const
{
  optix::Matrix4x4 T = optix::Matrix4x4::identity();
  T.setCol(3, m_matrix.getCol(3));
  return T;
}

void Transform::SetTranslation(float x, float y, float z)
{
  SetTranslation(make_float3(x, y, z));
}

void Transform::SetTranslation(const float3& translation)
{
  m_matrix.setCol(3, make_float4(translation, 1));
}

optix::Matrix4x4 Transform::GetScaleMatrix() const
{
  const optix::Matrix4x4 R = GetRotationMatrix();
  return R.inverse() * m_matrix;
}

float3 Transform::GetScale() const
{
  const optix::Matrix4x4 S = GetScaleMatrix();
  return make_float3(S[0], S[5], S[10]);
}

void Transform::SetScale(float x, float y, float z)
{
  SetScale(make_float3(x, y, z));
}

void Transform::SetScale(const float3& scale)
{
  const optix::Matrix4x4 T = GetTranslationMatrix();
  const optix::Matrix4x4 R = GetRotationMatrix();
  const optix::Matrix4x4 S = optix::Matrix4x4::scale(scale);
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