#pragma once

#include <optixu/optixpp.h>
#include <optixu/optixu_matrix.h>

namespace torch
{

class Transform
{
  public:

    Transform();

    ~Transform();

    optix::Matrix4x4 GetRotationMatrix() const;

    float3 GetEulerAngles() const;

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

    void Write(optix::Variable variable) const;

  protected:

    static float GetNorm(const optix::Matrix4x4& a, const optix::Matrix4x4& b);

  protected:

    optix::Matrix4x4 m_matrix;
};

} // namespace torch