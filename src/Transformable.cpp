#include <torch/Transformable.h>

namespace torch
{

Transformable::Transformable()
{
}

Transformable::~Transformable()
{
}

Transform Transformable::GetTransform() const
{
  return m_transform;
}

void Transformable::SetTransform(const Transform& transform)
{
  m_transform = transform;
  UpdateTransform();
}

float3 Transformable::GetOrientation() const
{
  return m_transform.GetEulerAngles();
}

void Transformable::SetOrientation(float x, float y, float z)
{
  m_transform.SetRotation(x, y, z);
  UpdateTransform();
}

float3 Transformable::GetPosition() const
{
  return m_transform.GetTranslation();
}

void Transformable::SetPosition(float x, float y, float z)
{
  m_transform.SetTranslation(x, y, z);
  UpdateTransform();
}

float3 Transformable::GetScale() const
{
  return m_transform.GetScale();
}

void Transformable::SetScale(float x, float y, float z)
{
  m_transform.SetScale(x, y, z);
  UpdateTransform();
}

void Transformable::SetScale(float scale)
{
  m_transform.SetScale(scale);
  UpdateTransform();
}

} // namespace torch