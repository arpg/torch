#pragma once

#include <torch/Transform.h>

namespace torch
{

class Transformable
{
  public:

    Transformable();

    Transform GetTransform() const;

    void SetTransform(const Transform& transform);

    float3 GetOrientation() const;

    void SetOrientation(float x, float y, float z);

    float3 GetPosition() const;

    void SetPosition(float x, float y, float z);

    float3 GetScale() const;

    void SetScale(float x, float y, float z);

    void SetScale(float scale);

  protected:

    virtual void UpdateTransform() = 0;

  protected:

    Transform m_transform;
};

} // namespace torch