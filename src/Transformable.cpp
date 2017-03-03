#include <torch/Transformable.h>
#include <torch/Vector.h>

namespace torch
{

Transformable::Transformable()
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

Vector Transformable::GetOrientation() const
{
  return m_transform.GetEulerAngles();
}

void Transformable::SetOrientation(float x, float y, float z, float w)
{
  m_transform.SetRotation(x, y, z, w);
  UpdateTransform();
}

void Transformable::SetOrientation(float x, float y, float z)
{
  m_transform.SetRotation(x, y, z);
  UpdateTransform();
}

Vector Transformable::GetPosition() const
{
  return m_transform.GetTranslation();
}

void Transformable::SetPosition(float x, float y, float z)
{
  m_transform.SetTranslation(x, y, z);
  UpdateTransform();
}

Vector Transformable::GetScale() const
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