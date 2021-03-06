#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <optixu/optixpp.h>
#include <optixu/optixu_matrix.h>

namespace torch
{

const float infinity = std::numeric_limits<float>::infinity();

class AlbedoResidualBlock;
class AreaLight;
class AreaLightData;
class AreaLightSampler;
class BoundingBox;
class Camera;
class CameraData;
class Context;
class Core;
class DirectionalLight;
class DirectionalLightData;
class DirectionalLightSampler;
class Distribution1D;
class Distribution2D;
class EnvironmentLight;
class EnvironmentLightData;
class EnvironmentLightSampler;
class Exception;
class Geometry;
class GeometryData;
class GeometryGroup;
class GeometryGroupSampler;
class GeometryGroupData;
class GeometrySampler;
class Group;
class Image;
class Light;
class LightData;
class LightSampler;
class Link;
class Material;
class MatteMaterial;
class Mesh;
class MeshData;
class MeshLoader;
class MeshSampler;
class Node;
class Normal;
class Object;
class Point;
class PointLight;
class PointLightData;
class PointLightSampler;
class Primitive;
class Random;
class Ray;
class Keyframe;
class KeyframeSet;
class Scene;
class SceneGeometrySampler;
class SceneLightSampler;
class SingleGeometry;
class SparseMatrix;
class Spectrum;
class Sphere;
class SphereData;
class SphereSampler;
class Transform;
class Transformable;
class Vector;
class VoxelLight;
class VoxelLightData;

} // namespace torch