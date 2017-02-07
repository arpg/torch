#include <torch/SceneLoader.h>
#include <sstream>
#include <tinyxml2.h>
#include <torch/AreaLight.h>
#include <torch/Camera.h>
#include <torch/DirectionalLight.h>
#include <torch/EnvironmentLight.h>
#include <torch/Exception.h>
#include <torch/GeometryGroup.h>
#include <torch/Group.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/PointLight.h>
#include <torch/Primitive.h>
#include <torch/Scene.h>
#include <torch/Spectrum.h>
#include <torch/Sphere.h>
#include <torch/Transform.h>
#include <torch/Vector.h>

namespace torch
{

SceneLoader::SceneLoader(const std::string& file) :
  m_file(file)
{
  Initialize();
}

void SceneLoader::Load(Scene& scene)
{
  m_scene = &scene;
  Element* elem = m_doc.FirstChildElement("scene");
  TORCH_ASSERT(elem, "scene file missing scene element");
  ParseScene(elem);
}

void SceneLoader::ParseScene(Element* elem)
{
  Element* childElem = elem->FirstChildElement();

  while (childElem)
  {
    std::shared_ptr<Node> node = ParseNode(childElem);
    childElem = childElem->NextSiblingElement();
    m_scene->Add(node);
  }
}

std::shared_ptr<Node> SceneLoader::ParseNode(Element* elem)
{
  const std::string type(elem->Value());
  TORCH_ASSERT(IsNode(elem), "invalid node type: " + type);
  std::shared_ptr<Node> node =(this->*m_nodeParsers[type])(elem);
  Element* childElem = elem->FirstChildElement();
  node->SetTransform(ParseTransform(elem));

  while (childElem)
  {
    if (IsNode(childElem))
    {
      const std::string type(childElem->Value());
      node->AddChild(ParseNode(childElem));
    }

    childElem = childElem->NextSiblingElement();
  }

  return node;
}

std::shared_ptr<Node> SceneLoader::ParseCamera(Element* elem)
{
  std::shared_ptr<Camera> camera;
  camera = m_scene->CreateCamera();

  unsigned int imageWidth, imageHeight;
  Element* imageSize = elem->FirstChildElement("image_size");
  TORCH_ASSERT(imageSize, "camera missing image_size");
  ParseUint2(imageSize, imageWidth, imageHeight);
  camera->SetImageSize(imageWidth, imageHeight);

  float fx, fy;
  Element* focalLength = elem->FirstChildElement("focal_length");
  TORCH_ASSERT(focalLength, "camera missing focal_length");
  ParseFloat2(focalLength, fx, fy);
  camera->SetFocalLength(fx, fy);

  float cx, cy;
  Element* centerPoint = elem->FirstChildElement("center_point");
  TORCH_ASSERT(centerPoint, "camera missing center_point");
  ParseFloat2(centerPoint, cx, cy);
  camera->SetCenterPoint(cx, cy);

  Element* sampleCount = elem->FirstChildElement("sample_count");
  TORCH_ASSERT(sampleCount, "camera missing sample_count");
  camera->SetSampleCount(ParseUnsignedInt(sampleCount));

  return camera;
}

std::shared_ptr<Node> SceneLoader::ParseGroup(Element* elem)
{
  return m_scene->CreateGroup();
}

std::shared_ptr<Node> SceneLoader::ParseAreaLight(Element* elem)
{
  std::shared_ptr<AreaLight> light;
  light = m_scene->CreateAreaLight();
  light->SetGeometry(ParseGeometry(elem));
  Element* radiance = elem->FirstChildElement("radiance");
  if (radiance) light->SetRadiance(ParseSpectrum(radiance));
  return light;
}

std::shared_ptr<Node> SceneLoader::ParseDirectionalLight(Element* elem)
{
  std::shared_ptr<DirectionalLight> light;
  light->SetTransform(ParseTransform(elem));
  Element* radiance = elem->FirstChildElement("radiance");
  if (radiance) light->SetRadiance(ParseSpectrum(radiance));
  Element* direction = elem->FirstChildElement("direction");
  if (direction) light->SetDirection(ParseVector(direction));
  return light;
}

std::shared_ptr<Node> SceneLoader::ParseEnvironmentLight(Element* elem)
{
  std::shared_ptr<EnvironmentLight> light;
  light = m_scene->CreateEnvironmentLight();
  Element* rowCount = elem->FirstChildElement("row_count");
  TORCH_ASSERT(rowCount, "environment_light missing row_count");
  light->SetRowCount(ParseUnsignedInt(rowCount));
  Element* radiance = elem->FirstChildElement("radiance");
  if (radiance) light->SetRadiance(ParseSpectrum(radiance));
  return light;
}

std::shared_ptr<Node> SceneLoader::ParsePointLight(Element* elem)
{
  std::shared_ptr<PointLight> light;
  light = m_scene->CreatePointLight();
  Element* intensity = elem->FirstChildElement("intensity");
  if (intensity) light->SetIntensity(ParseSpectrum(intensity));
  return light;
}

std::shared_ptr<Node> SceneLoader::ParsePrimitive(Element* elem)
{
  std::shared_ptr<Primitive> primitive;
  primitive = m_scene->CreatePrimitive();
  primitive->SetGeometry(ParseGeometry(elem));
  primitive->SetMaterial(ParseMaterial(elem));
  return primitive;
}

std::shared_ptr<Geometry> SceneLoader::ParseGeometry(Element* elem)
{
  Element* childElem = elem->FirstChildElement();

  while (childElem)
  {
    if (IsGeometry(childElem))
    {
      const std::string type(childElem->Value());

      std::shared_ptr<Geometry> geometry =
          (this->*m_geometryParsers[type])(childElem);

      geometry->SetTransform(ParseTransform(childElem));
      return geometry;
    }

    childElem = childElem->NextSiblingElement();
  }

  throw Exception(std::string(elem->Value()) + " missing geometry");
}

std::shared_ptr<Geometry> SceneLoader::ParseGeometryGroup(Element* elem)
{
  std::shared_ptr<GeometryGroup> group = m_scene->CreateGeometryGroup();
  Element* childElem = elem->FirstChildElement();

  while (childElem)
  {
    if (IsGeometry(childElem))
    {
      const std::string type(childElem->Value());
      group->AddChild((this->*m_geometryParsers[type])(childElem));
    }

    childElem = childElem->NextSiblingElement();
  }

  return group;
}

std::shared_ptr<Geometry> SceneLoader::ParseMesh(Element* elem)
{
  Element* file = elem->FirstChildElement("file");
  TORCH_ASSERT(file, "mesh missing file");
  std::shared_ptr<Mesh> mesh = m_scene->CreateMesh(file->GetText());
  return mesh;
}

std::shared_ptr<Geometry> SceneLoader::ParseSphere(Element* elem)
{
  return m_scene->CreateSphere();
}

std::shared_ptr<Material> SceneLoader::ParseMaterial(Element* elem)
{
  Element* childElem = elem->FirstChildElement();

  while (childElem)
  {
    if (IsMaterial(childElem))
    {
      const std::string type(childElem->Value());
      return (this->*m_materialParsers[type])(childElem);
    }

    childElem = childElem->NextSiblingElement();
  }

  throw Exception(std::string(elem->Value()) + " missing material");
}

std::shared_ptr<Material> SceneLoader::ParseMatteMaterial(Element* elem)
{
  std::shared_ptr<MatteMaterial> material = m_scene->CreateMatteMaterial();
  Element* albedo = elem->FirstChildElement("albedo");
  TORCH_ASSERT(albedo, "matte_material missing albedo");
  material->SetAlbedo(ParseSpectrum(albedo));
  return material;
}

Spectrum SceneLoader::ParseSpectrum(Element* elem)
{
  return Spectrum::FromRGB(ParseVector(elem));
}

Transform SceneLoader::ParseTransform(Element* elem)
{
  Transform transform;
  transform.SetRotation(ParseOrientation(elem));
  transform.SetTranslation(ParsePosition(elem));
  transform.SetScale(ParseScale(elem));
  return transform;
}

Vector SceneLoader::ParseOrientation(Element* elem)
{
  Vector vector(0, 0, 0);
  elem = elem->FirstChildElement("orientation");
  if (elem) vector = ParseVector(elem);
  return vector;
}

Vector SceneLoader::ParsePosition(Element* elem)
{
  Vector vector(0, 0, 0);
  elem = elem->FirstChildElement("position");
  if (elem) vector = ParseVector(elem);
  return vector;
}

Vector SceneLoader::ParseScale(Element* elem)
{
  Vector vector(1, 1, 1);
  elem = elem->FirstChildElement("scale");
  if (elem) vector = ParseVector(elem);
  return vector;
}

Vector SceneLoader::ParseVector(Element* elem)
{
  Vector vector;
  const std::string value(elem->GetText());
  std::stringstream tokenizer(value);
  tokenizer >> vector.x;
  tokenizer >> vector.y;
  tokenizer >> vector.z;
  return vector;
}

unsigned int SceneLoader::ParseUnsignedInt(Element* elem)
{
  unsigned int result;
  const std::string value(elem->GetText());
  std::stringstream tokenizer(value);
  tokenizer >> result;
  return result;
}

void SceneLoader::ParseUint2(SceneLoader::Element* elem,
    unsigned int& x, unsigned int& y)
{
  const std::string value(elem->GetText());
  std::stringstream tokenizer(value);
  tokenizer >> x;
  tokenizer >> y;
}

void SceneLoader::ParseFloat2(SceneLoader::Element* elem, float& x, float& y)
{
  const std::string value(elem->GetText());
  std::stringstream tokenizer(value);
  tokenizer >> x;
  tokenizer >> y;
}

bool SceneLoader::IsNode(SceneLoader::Element* elem)
{
  auto iter = m_nodeParsers.find(elem->Value());
  return iter != m_nodeParsers.end();
}

bool SceneLoader::IsGeometry(SceneLoader::Element* elem)
{
  auto iter = m_geometryParsers.find(elem->Value());
  return iter != m_geometryParsers.end();
}

bool SceneLoader::IsMaterial(SceneLoader::Element* elem)
{
  auto iter = m_materialParsers.find(elem->Value());
  return iter != m_materialParsers.end();
}

void SceneLoader::Initialize()
{
  LoadFile();
  CreateNodeParsers();
  CreateGeometryParsers();
  CreateMaterialParsers();
}

void SceneLoader::LoadFile()
{
  m_doc.LoadFile(m_file.c_str());
  TORCH_ASSERT(!m_doc.Error(), "unable to read scene xml file");
}

void SceneLoader::CreateNodeParsers()
{
  m_nodeParsers["camera"] = &SceneLoader::ParseCamera;
  m_nodeParsers["area_light"] = &SceneLoader::ParseAreaLight;
  m_nodeParsers["directional_light"] = &SceneLoader::ParseDirectionalLight;
  m_nodeParsers["environment_light"] = &SceneLoader::ParseEnvironmentLight;
  m_nodeParsers["point_light"] = &SceneLoader::ParsePointLight;
  m_nodeParsers["primitive"] = &SceneLoader::ParsePrimitive;
}

void SceneLoader::CreateGeometryParsers()
{
  m_geometryParsers["geometry_group"] = &SceneLoader::ParseGeometryGroup;
  m_geometryParsers["mesh"] = &SceneLoader::ParseMesh;
  m_geometryParsers["sphere"] = &SceneLoader::ParseSphere;
}

void SceneLoader::CreateMaterialParsers()
{
  m_materialParsers["matte_material"] = &SceneLoader::ParseMatteMaterial;
}

} // namespace torch