#pragma once

#include <torch/Core.h>

namespace torch
{

class Normal;
class Point;
class Vector;

class Transform
{
  public:

    Transform();

    Transform(const optix::Matrix4x4& matrix);

    Transform(optix::Transform transform);

    optix::Matrix4x4 GetRotationMatrix() const;

    float3 GetEulerAngles() const;

    void SetRotation(const optix::Matrix3x3& rotation);

    void SetRotation(float x, float y, float z);

    void SetRotation(const float3& rotation);

    float3 GetTranslation() const;

    optix::Matrix4x4 GetTranslationMatrix() const;

    void SetTranslation(float x, float y, float z);

    void SetTranslation(const float3& translation);

    optix::Matrix4x4 GetScaleMatrix() const;

    float3 GetScale() const;

    void SetScale(float x, float y, float z);

    void SetScale(const float3& scale);

    void SetScale(float scale);

    optix::Matrix4x4 GetMatrix() const;

    optix::Matrix3x4 GetMatrix3x4() const;

    Transform Inverse() const;

    Transform operator*(const Transform& transform) const;

    Normal operator*(const Normal& normal) const;

    Point operator*(const Point& point) const;

    Vector operator*(const Vector& vector) const;

    BoundingBox operator*(const BoundingBox& b) const;

    void Write(optix::Transform transform) const;

    void Write(optix::Variable variable) const;

  protected:

    static float GetNorm(const optix::Matrix4x4& a, const optix::Matrix4x4& b);

  protected:

    optix::Matrix4x4 m_matrix;
};

} // namespace torch