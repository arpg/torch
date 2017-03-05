#pragma once

#include <sophus/se3.hpp>
#include <torch/Transform.h>

namespace torch
{

class Transformable
{
  public:

    Transformable();

    Transform GetTransform() const;

    void SetTransform(const Transform& transform);

    void SetTransform(const Sophus::SE3f& transform);

    Vector GetOrientation() const;

    void SetOrientation(float x, float y, float z, float w);

    void SetOrientation(float x, float y, float z);

    Vector GetPosition() const;

    void SetPosition(float x, float y, float z);

    Vector GetScale() const;

    void SetScale(float x, float y, float z);

    void SetScale(float scale);

  protected:

    virtual void UpdateTransform() = 0;

  protected:

    Transform m_transform;
};

} // namespace torch