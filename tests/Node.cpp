#include <gtest/gtest.h>
#include <torch/Torch.h>

using namespace torch;

TEST(Node, Children)
{
  Scene scene;
  std::shared_ptr<Camera> parent;
  std::shared_ptr<Camera> child1;
  std::shared_ptr<Camera> child2;
  std::shared_ptr<Camera> child3;

  parent = scene.CreateCamera();
  child1 = scene.CreateCamera();
  child2 = scene.CreateCamera();
  child3 = scene.CreateCamera();

  ASSERT_EQ(0, parent->GetChildCount());
  ASSERT_EQ(false,  parent->HasChild(parent));
  ASSERT_EQ(false,  parent->HasChild(child1));
  ASSERT_EQ(false,  parent->HasChild(child2));
  ASSERT_EQ(false,  parent->HasChild(child3));

  parent->AddChild(parent);
  ASSERT_EQ(0, parent->GetChildCount());
  ASSERT_EQ(false,  parent->HasChild(parent));
  ASSERT_EQ(false,  parent->HasChild(child1));
  ASSERT_EQ(false,  parent->HasChild(child2));
  ASSERT_EQ(false,  parent->HasChild(child3));

  parent->AddChild(child1);
  ASSERT_EQ(1, parent->GetChildCount());
  ASSERT_EQ(false,  parent->HasChild(parent));
  ASSERT_EQ(true,   parent->HasChild(child1));
  ASSERT_EQ(false,  parent->HasChild(child2));
  ASSERT_EQ(false,  parent->HasChild(child3));
  ASSERT_EQ(child1, parent->GetChild(0));

  parent->AddChild(child2);
  ASSERT_EQ(2, parent->GetChildCount());
  ASSERT_EQ(false,  parent->HasChild(parent));
  ASSERT_EQ(true,   parent->HasChild(child1));
  ASSERT_EQ(true,   parent->HasChild(child2));
  ASSERT_EQ(false,  parent->HasChild(child3));
  ASSERT_EQ(child1, parent->GetChild(0));
  ASSERT_EQ(child2, parent->GetChild(1));

  parent->AddChild(child3);
  ASSERT_EQ(3, parent->GetChildCount());
  ASSERT_EQ(false,  parent->HasChild(parent));
  ASSERT_EQ(true,   parent->HasChild(child1));
  ASSERT_EQ(true,   parent->HasChild(child2));
  ASSERT_EQ(true,   parent->HasChild(child3));
  ASSERT_EQ(child1, parent->GetChild(0));
  ASSERT_EQ(child2, parent->GetChild(1));
  ASSERT_EQ(child3, parent->GetChild(2));

  parent->RemoveChild(child2);
  ASSERT_EQ(2, parent->GetChildCount());
  ASSERT_EQ(false,  parent->HasChild(parent));
  ASSERT_EQ(true,   parent->HasChild(child1));
  ASSERT_EQ(false,  parent->HasChild(child2));
  ASSERT_EQ(true,   parent->HasChild(child3));
  ASSERT_EQ(child1, parent->GetChild(0));
  ASSERT_EQ(child3, parent->GetChild(1));

  parent->RemoveChildren();
  ASSERT_EQ(0, parent->GetChildCount());
  ASSERT_EQ(false,  parent->HasChild(parent));
  ASSERT_EQ(false,  parent->HasChild(child1));
  ASSERT_EQ(false,  parent->HasChild(child2));
  ASSERT_EQ(false,  parent->HasChild(child3));
}