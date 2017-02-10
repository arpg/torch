#include <Eigen/Eigen>
#include <gtest/gtest.h>
#include <torch/Torch.h>

using namespace torch;

const float epsilon = 1E-4;

Eigen::Matrix4f GetRotationMatrix(const Vector& rotation)
{
  const Eigen::Quaternionf quaternion =
      Eigen::AngleAxisf(rotation.x, Eigen::Vector3f::UnitX()) *
      Eigen::AngleAxisf(rotation.y, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(rotation.z, Eigen::Vector3f::UnitZ());

  Eigen::Matrix4f result(Eigen::Matrix4f::Identity());
  result.block<3, 3>(0, 0) = Eigen::Matrix3f(quaternion);
  return result;
}

Eigen::Matrix4f GetRotationMatrix(const optix::Matrix4x4& rotation)
{
  optix::Matrix4x4 transpose = rotation.transpose();
  typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> RowMatrix;
  return Eigen::Map<RowMatrix>(transpose.getData());
}

TEST(Transform, Translation)
{
  Transform transform;
  Vector expected;
  Vector found;

  expected = Vector(0, 0, 0);
  found = transform.GetTranslation();
  ASSERT_NEAR(expected.x, found.x, epsilon);
  ASSERT_NEAR(expected.y, found.y, epsilon);
  ASSERT_NEAR(expected.z, found.z, epsilon);

  expected = Vector(1, 2, 3);
  transform.SetTranslation(expected);
  found = transform.GetTranslation();
  ASSERT_NEAR(expected.x, found.x, epsilon);
  ASSERT_NEAR(expected.y, found.y, epsilon);
  ASSERT_NEAR(expected.z, found.z, epsilon);

  expected = Vector(0, -1, 2);
  transform.SetTranslation(expected);
  found = transform.GetTranslation();
  ASSERT_NEAR(expected.x, found.x, epsilon);
  ASSERT_NEAR(expected.y, found.y, epsilon);
  ASSERT_NEAR(expected.z, found.z, epsilon);

  expected = Vector(0, 0, 0);
  transform.SetTranslation(expected);
  found = transform.GetTranslation();
  ASSERT_NEAR(expected.x, found.x, epsilon);
  ASSERT_NEAR(expected.y, found.y, epsilon);
  ASSERT_NEAR(expected.z, found.z, epsilon);
}

TEST(Transform, Rotation)
{
  Transform transform;
  Eigen::Matrix4f expected;
  Eigen::Matrix4f found;
  Vector rotation;

  rotation = Vector(0, 0, 0);
  expected = GetRotationMatrix(rotation);
  found = GetRotationMatrix(transform.GetRotationMatrix());

  for (unsigned int i = 0; i < 16; ++i)
  {
    ASSERT_NEAR(expected.data()[i], found.data()[i], epsilon);
  }

  rotation = Vector(0.5, 0, 0);
  transform.SetRotation(rotation);
  expected = GetRotationMatrix(rotation);
  found = GetRotationMatrix(transform.GetRotationMatrix());

  for (unsigned int i = 0; i < 16; ++i)
  {
    ASSERT_NEAR(expected.data()[i], found.data()[i], epsilon);
  }

  rotation = Vector(0, 0.5, 0);
  transform.SetRotation(rotation);
  expected = GetRotationMatrix(rotation);
  found = GetRotationMatrix(transform.GetRotationMatrix());

  for (unsigned int i = 0; i < 16; ++i)
  {
    ASSERT_NEAR(expected.data()[i], found.data()[i], epsilon);
  }

  rotation = Vector(0, 0, 0.5);
  transform.SetRotation(rotation);
  expected = GetRotationMatrix(rotation);
  found = GetRotationMatrix(transform.GetRotationMatrix());

  for (unsigned int i = 0; i < 16; ++i)
  {
    ASSERT_NEAR(expected.data()[i], found.data()[i], epsilon);
  }

  rotation = Vector(0.1, 0.2, 0.3);
  transform.SetRotation(rotation);
  expected = GetRotationMatrix(rotation);
  found = GetRotationMatrix(transform.GetRotationMatrix());

  for (unsigned int i = 0; i < 16; ++i)
  {
    ASSERT_NEAR(expected.data()[i], found.data()[i], epsilon);
  }

  rotation = Vector(1, 2, 3);
  transform.SetRotation(rotation);
  expected = GetRotationMatrix(rotation);
  found = GetRotationMatrix(transform.GetRotationMatrix());

  for (unsigned int i = 0; i < 16; ++i)
  {
    ASSERT_NEAR(expected.data()[i], found.data()[i], epsilon);
  }

  rotation = Vector(-1, 2, -3);
  transform.SetRotation(rotation);
  expected = GetRotationMatrix(rotation);
  found = GetRotationMatrix(transform.GetRotationMatrix());

  for (unsigned int i = 0; i < 16; ++i)
  {
    ASSERT_NEAR(expected.data()[i], found.data()[i], epsilon);
  }

  rotation = Vector(0.1, 0.2, 0.3);
  transform.SetScale(1, 2, 3);
  transform.SetRotation(rotation);
  expected = GetRotationMatrix(rotation);
  found = GetRotationMatrix(transform.GetRotationMatrix());

  for (unsigned int i = 0; i < 16; ++i)
  {
    ASSERT_NEAR(expected.data()[i], found.data()[i], epsilon);
  }

  rotation = Vector(-1, 2, -3);
  transform.SetScale(0.1, 0.2, 0.3);
  transform.SetRotation(rotation);
  expected = GetRotationMatrix(rotation);
  found = GetRotationMatrix(transform.GetRotationMatrix());

  for (unsigned int i = 0; i < 16; ++i)
  {
    ASSERT_NEAR(expected.data()[i], found.data()[i], epsilon);
  }
}

TEST(Transform, Scale)
{
  Transform transform;
  Vector expected;
  Vector found;

  expected = Vector(1, 1, 1);
  found = transform.GetScale();
  ASSERT_NEAR(expected.x, found.x, epsilon);
  ASSERT_NEAR(expected.y, found.y, epsilon);
  ASSERT_NEAR(expected.z, found.z, epsilon);

  expected = Vector(1, 2, 3);
  transform.SetScale(expected);
  found = transform.GetScale();
  ASSERT_NEAR(expected.x, found.x, epsilon);
  ASSERT_NEAR(expected.y, found.y, epsilon);
  ASSERT_NEAR(expected.z, found.z, epsilon);

  expected = Vector(2, 2, 2);
  transform.SetScale(expected);
  found = transform.GetScale();
  ASSERT_NEAR(expected.x, found.x, epsilon);
  ASSERT_NEAR(expected.y, found.y, epsilon);
  ASSERT_NEAR(expected.z, found.z, epsilon);

  expected = Vector(0.1, 0.2, 0.3);
  transform.SetScale(expected);
  found = transform.GetScale();
  ASSERT_NEAR(expected.x, found.x, epsilon);
  ASSERT_NEAR(expected.y, found.y, epsilon);
  ASSERT_NEAR(expected.z, found.z, epsilon);

  transform.SetRotation(0.1, 0.2, 0.3);
  expected = Vector(0.1, 0.2, 0.3);
  transform.SetScale(expected);
  found = transform.GetScale();
  ASSERT_NEAR(expected.x, found.x, epsilon);
  ASSERT_NEAR(expected.y, found.y, epsilon);
  ASSERT_NEAR(expected.z, found.z, epsilon);

  transform.SetRotation(0.3, 0.2, 0.1);
  expected = Vector(1, 2, 3);
  transform.SetScale(expected);
  found = transform.GetScale();
  ASSERT_NEAR(expected.x, found.x, epsilon);
  ASSERT_NEAR(expected.y, found.y, epsilon);
  ASSERT_NEAR(expected.z, found.z, epsilon);
}