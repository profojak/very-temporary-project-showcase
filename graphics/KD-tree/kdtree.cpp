#include "kdtree.h"
 
#include <algorithm>

extern int
triBoxOverlap(float boxcenter[3], float boxhalfsize[3], float triverts[3][3]);
 
CKDTree::CKDTree():
  maxDepth(24), maxLeafSize(4), useBinning(false), useDrivingAxis(true),
  bins(8), inverseBins(1.0f / static_cast<float>(bins)), useClipping(false),
  usePruning(false)
{
  objlist = nullptr;
  builtUp = false;
  stack.reserve(maxDepth);
}
 
CKDTree::CKDTree(const int maxDepth, const int maxLeafSize,
  const bool useBinning, const bool useDrivingAxis, const int bins,
  const bool useClipping, const bool usePruning):
  maxDepth(maxDepth), maxLeafSize(static_cast<unsigned>(maxLeafSize)),
  useBinning(useBinning), useDrivingAxis(useDrivingAxis), bins(bins),
  inverseBins(1.0f / static_cast<float>(bins)),
  useClipping(useClipping), usePruning(usePruning)
{
  objlist = nullptr;
  builtUp = false;
  stack.reserve(maxDepth);
  if (useBinning) {
    binObjectStarts.resize(bins);
    binObjectEnds.resize(bins);
  }
}
 
CKDTree::~CKDTree()
{
  Remove();
}
 
void
CKDTree::BuildUp(const CObjectList &objectList)
{
  objlist = const_cast<CObjectList*>(&objectList);
  bbox.Initialize();
  InitializeBox(bbox, *objlist);
 
  nodes.reserve(objlist->size());
 
  // Conservative estimate of the number of nodes and object references
  references.reserve(3 * objlist->size());
 
  // All empty leaf nodes point to dummy reference 0
  references.push_back(0);
 
  vector<unsigned> currentReferences(objlist->size());
  for (unsigned i = 0; i < objlist->size(); i++)
    currentReferences[i] = i;
 
  currentDepth = 0;
  BuildRecursively(currentReferences, bbox);
  ratioEmptyLeaves /= numLeafNodes;
  avgLeafDepth /= numLeafNodes;
  avgTriangles = static_cast<float>(numTriangles) / numLeafNodes;
  builtUp = true;
}
 
void
CKDTree::Remove()
{
  objlist = nullptr;
}
 
void
CKDTree::ProvideID(ostream &app)
{
  app << "#CASDS = CKDTree - binning clipping pruning kd-tree\n";
  app << static_cast<int>(GetID()) << "\n";
}
 
void
CKDTree::PrintStatistics()
{
  STATUS << "numIncidenceTests = " << numIncidenceTests << endl;
  STATUS << "numTraversalSteps = " << numTraversalSteps << endl;
  STATUS << "numInternalNodes = " << numInternalNodes << endl;
  STATUS << "numLeafNodes = " << numLeafNodes << endl;
  STATUS << "ratioEmptyLeaves = " << ratioEmptyLeaves << endl;
  STATUS << "maxLeafDepth = " << maxLeafDepth << endl;
  STATUS << "avgLeafDepth = " << avgLeafDepth << endl;
  STATUS << "numTriangles = " << numTriangles << endl;
  STATUS << "avgTriangles = " << avgTriangles << endl;
  STATUS << "numClippedTriangles = " << numClippedTriangles << endl;
  STATUS << "numPrunedTriangles = " << numPrunedTriangles << endl;
  STATUS << "memory ~= " << nodes.size() * sizeof(SKDNode) +
    references.size() * sizeof(unsigned) +
    clippedReferences.size() * sizeof(pair<unsigned, SBBox>) +
    objlist->size() * sizeof(CObject3D*) << " bytes" << endl;
}
 
SBBox
CKDTree::Union(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c)
{
  SBBox bbox;
  bbox.SetMin(min(a.x, min(b.x, c.x)), min(a.y, min(b.y, c.y)),
    min(a.z, min(b.z, c.z)));
  bbox.SetMax(max(a.x, max(b.x, c.x)), max(a.y, max(b.y, c.y)),
    max(a.z, max(b.z, c.z)));
  return bbox;
}
 
SBBox
CKDTree::Union(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c,
  const glm::vec3 &d)
{
  SBBox bbox;
  bbox.SetMin(min(a.x, min(b.x, min(c.x, d.x))),
    min(a.y, min(b.y, min(c.y, d.y))),
    min(a.z, min(b.z, min(c.z, d.z))));
  bbox.SetMax(max(a.x, max(b.x, max(c.x, d.x))),
    max(a.y, max(b.y, max(c.y, d.y))),
    max(a.z, max(b.z, max(c.z, d.z))));
  return bbox;
}
 
void
CKDTree::BuildRecursively(vector<unsigned> ¤tReferences,
  const SBBox ¤tBBox)
{
  const unsigned index = nodes.size();
  nodes.emplace_back();
 
  // Create leaf node if conditions are met
  if (currentReferences.size() <= maxLeafSize || currentDepth >= maxDepth) {
    CreateLeaf(index, currentReferences, currentBBox);
    numLeafNodes++;
    return;
  }
 
  numInternalNodes++;
 
  // Find split position and split bounding box
  unsigned rightCount = currentReferences.size();
  if (useBinning)
    SAHBinningSplit(index, currentReferences, currentBBox, rightCount);
  else
    SpatialMedianSplit(index, currentBBox);
 
  SBBox leftBBox(currentBBox), rightBBox(currentBBox);
  leftBBox.SetMax(nodes[index].GetAxis(), nodes[index].split);
  rightBBox.SetMin(nodes[index].GetAxis(), nodes[index].split);
 
  // Partition references and create child nodes
  vector<unsigned> rightReferences;
  rightReferences.reserve(rightCount);
  PartitionReferences(currentReferences, rightReferences, leftBBox, rightBBox,
    nodes[index].GetAxis(), nodes[index].split);
 
  CreateChildren(index, currentReferences, rightReferences,
    leftBBox, rightBBox);
}
 
void
CKDTree::CreateLeaf(const unsigned index,
  const vector<unsigned> ¤tReferences, const SBBox ¤tBBox)
{
  if (currentDepth > maxLeafDepth)
    maxLeafDepth = currentDepth;
  avgLeafDepth += currentDepth;
 
  nodes[index].MarkAsLeaf();
  if (currentReferences.empty()) {
    ratioEmptyLeaves += 1.0f;
    return;
  }
 
  // Prune objects that do not overlap the bounding box
  if (usePruning) {
    unsigned count = 0;
    const unsigned objectStarts = references.size();
    auto center = 0.5f * (currentBBox.Min() + currentBBox.Max());
    auto halfSize = 0.5f * currentBBox.Diagonal();
 
    for (const unsigned reference : currentReferences) {
      const auto &triangle = dynamic_cast<CTriangle*>((*objlist)[
        GetObjectReference(reference)]);
      if (triBoxOverlap(reinterpret_cast<float*>(¢er),
        reinterpret_cast<float*>(&halfSize),
        reinterpret_cast<float (*)[3]>(triangle->vertices))) {
        if (count == 0)
          references.push_back(0);
        references.push_back(reference);
        count++;
      } else
        numPrunedTriangles++;
    }
    if (count != 0) {
      nodes[index].SetReference(objectStarts);
      numTriangles += static_cast<int>(count);
      references[objectStarts] = count;
    } else
      ratioEmptyLeaves += 1.0f;
 
  } else {
    nodes[index].SetReference(references.size());
    numTriangles += static_cast<int>(currentReferences.size());
    references.push_back(currentReferences.size());
    for (const unsigned reference : currentReferences)
      references.push_back(reference);
  }
}
 
void
CKDTree::SpatialMedianSplit(const unsigned index, const SBBox ¤tBBox)
{
  unsigned axis = DrivingAxis(currentBBox.Diagonal());
  nodes[index].SetAxis(axis);
  nodes[index].split = 0.5f * (currentBBox.Min(axis) + currentBBox.Max(axis));
}
 
void
CKDTree::SAHBinningSplit(const unsigned index,
  const vector<unsigned> ¤tReferences, const SBBox ¤tBBox,
  unsigned &rightCount)
{
  SBBox leftBBox, rightBBox;
  float bestCost = CLimits::Infinity;
 
  // Consider only the longest axis
  if (useDrivingAxis) {
    const unsigned axis = DrivingAxis(currentBBox.Diagonal());
    BinReferences(currentReferences, axis, currentBBox);
    leftBBox = currentBBox;
    rightBBox = currentBBox;
    FindBinningCost(index, currentReferences, axis, currentBBox,
      leftBBox, rightBBox, rightCount, bestCost);
 
  // Consider all axes
  } else {
    for (unsigned axis = 0; axis < 3; axis++) {
      BinReferences(currentReferences, axis, currentBBox);
      leftBBox = currentBBox;
      rightBBox = currentBBox;
      FindBinningCost(index, currentReferences, axis, currentBBox,
        leftBBox, rightBBox, rightCount, bestCost);
    }
  }
}
 
void
CKDTree::BinReferences(const vector<unsigned> ¤tReferences,
  const unsigned axis, const SBBox ¤tBBox)
{
  binObjectStarts.assign(bins, 0);
  binObjectEnds.assign(bins, 0);
  const float range = static_cast<float>(bins) /
    (currentBBox.Max(axis) - currentBBox.Min(axis));
  for (const unsigned reference : currentReferences) {
    const auto &objectBBox = GetObjectBBox(reference);
    const int binStart = max(0, min(bins - 1, static_cast<int>(
      range * (objectBBox.Min(axis) - currentBBox.Min(axis)))));
    const int binEnd = max(0, min(bins - 1, static_cast<int>(
      range * (objectBBox.Max(axis) - currentBBox.Min(axis)))));
    binObjectStarts[binStart]++;
    binObjectEnds[binEnd]++;
  }
}
 
void
CKDTree::FindBinningCost(const unsigned index,
  const vector<unsigned> ¤tReferences, const unsigned axis,
  const SBBox ¤tBBox, SBBox &leftBBox, SBBox &rightBBox,
  unsigned &rightCount, float &bestCost)
{
  unsigned objectStarts = 0;
  unsigned objectEnds = currentReferences.size();
  const float step = (currentBBox.Max(axis) - currentBBox.Min(axis)) *
    inverseBins;
  for (int i = 0; i < bins - 1; i++) {
    const float split = currentBBox.Min(axis) + step * (i + 1);
    objectStarts += binObjectStarts[i];
    objectEnds -= binObjectEnds[i];
    leftBBox.SetMax(axis, split);
    rightBBox.SetMin(axis, split);
    const float cost = leftBBox.SA2() * objectStarts +
      rightBBox.SA2() * objectEnds;
    if (cost < bestCost) {
      bestCost = cost;
      nodes[index].split = split;
      nodes[index].SetAxis(axis);
      rightCount = objectEnds;
    }
  }
}
 
void
CKDTree::PartitionReferences(vector<unsigned> ¤tReferences,
  vector<unsigned> &rightReferences, const SBBox &leftBBox,
  const SBBox &rightBBox, const unsigned axis, const float split)
{
  // Clip references straddling the splitting plane if clipping is used
  if (useClipping) {
    unsigned leftCount = 0;
    bool overlapsLeft = false, overlapsRight = false;
    for (unsigned i = 0; i < currentReferences.size(); i++) {
      const auto &objectBBox = GetObjectBBox(currentReferences[i]);
      const unsigned reference = GetObjectReference(currentReferences[i]);
      overlapsLeft = OverlapS(leftBBox, objectBBox);
      overlapsRight = OverlapS(rightBBox, objectBBox);
 
      // If the object is a triangle and overlaps both spaces, clip it
      if (overlapsLeft && overlapsRight) {
        const auto &triangle = dynamic_cast<CTriangle*>((*objlist)[reference]);
        if (triangle) {
          numClippedTriangles++;
 
          // First, sort the vertices by the axis
          int indices[3] = {0, 1, 2};
          sort(indices, indices + 3, [&](const int a, const int b) {
            return triangle->vertices[a][axis] < triangle->vertices[b][axis];
          });
 
          // There are only five possible cases:
 
          // The triangle lies completely in the left space and only touches
          // the right space with the rightmost vertex
          // Vertex signs: -1 -1 0
          if (Sign(triangle->vertices[indices[2]][axis] - split) == 0)
            currentReferences[leftCount++] = reference;
 
          // The triangle lies completely in the right space and only touches
          // the left space with the leftmost vertex
          // Vertex signs: 0 1 1
          else if (Sign(triangle->vertices[indices[0]][axis] - split) == 0)
            rightReferences.push_back(reference);
 
          // The middle vertex lies on the splitting plane, only one
          // intersection must be found
          // Vertex signs: -1 0 1
          else if (Sign(triangle->vertices[indices[1]][axis] - split) == 0) {
            const glm::vec3 intersection = SplitPlaneIntersection(axis, split,
              triangle->vertices[indices[0]], triangle->vertices[indices[2]]);
            SBBox newLeftBBox = Union(triangle->vertices[indices[0]],
              triangle->vertices[indices[1]], intersection);
            SBBox newRightBBox = Union(intersection,
              triangle->vertices[indices[1]], triangle->vertices[indices[2]]);
 
            currentReferences[leftCount++] = objlist->size() +
              clippedReferences.size();
            clippedReferences.push_back({reference, newLeftBBox});
            rightReferences.push_back(objlist->size() +
              clippedReferences.size());
            clippedReferences.push_back({reference, newRightBBox});
 
          // The leftmost vertex lies in the left space and other two vertices
          // lie in the right space, two intersections must be found
          // Vertex signs: -1 1 1
          } else if (Sign(triangle->vertices[indices[0]][axis] - split) == -1 &&
            Sign(triangle->vertices[indices[1]][axis] - split) == 1) {
            const glm::vec3 intersection1 = SplitPlaneIntersection(axis, split,
              triangle->vertices[indices[0]], triangle->vertices[indices[2]]);
            const glm::vec3 intersection2 = SplitPlaneIntersection(axis, split,
              triangle->vertices[indices[0]], triangle->vertices[indices[1]]);
            SBBox newLeftBBox = Union(triangle->vertices[indices[0]],
              intersection1, intersection2);
            SBBox newRightBBox = Union(intersection1, intersection2,
              triangle->vertices[indices[1]], triangle->vertices[indices[2]]);
 
            if (usePruning) {
              if (OverlapS(leftBBox, newLeftBBox)) {
                currentReferences[leftCount++] = objlist->size() +
                  clippedReferences.size();
                clippedReferences.push_back({reference, newLeftBBox});
              } else
                numPrunedTriangles++;
              if (OverlapS(rightBBox, newRightBBox)) {
                rightReferences.push_back(objlist->size() +
                  clippedReferences.size());
                clippedReferences.push_back({reference, newRightBBox});
              } else
                numPrunedTriangles++;
            } else {
              currentReferences[leftCount++] = objlist->size() +
                clippedReferences.size();
              clippedReferences.push_back({reference, newLeftBBox});
              rightReferences.push_back(objlist->size() +
                clippedReferences.size());
              clippedReferences.push_back({reference, newRightBBox});
            }
 
          // The rightmost vertex lies in the right space and other two vertices
          // lie in the left space, two intersections must be found
          // Vertex signs: -1 -1 1
          } else if (Sign(triangle->vertices[indices[2]][axis] - split) == 1 &&
            Sign(triangle->vertices[indices[1]][axis] - split) == -1) {
            const glm::vec3 intersection1 = SplitPlaneIntersection(axis, split,
              triangle->vertices[indices[0]], triangle->vertices[indices[2]]);
            const glm::vec3 intersection2 = SplitPlaneIntersection(axis, split,
              triangle->vertices[indices[1]], triangle->vertices[indices[2]]);
            SBBox newLeftBBox = Union(triangle->vertices[indices[0]],
              triangle->vertices[indices[1]], intersection1, intersection2);
            SBBox newRightBBox = Union(intersection1, intersection2,
              triangle->vertices[indices[2]]);
 
            if (usePruning) {
              if (OverlapS(leftBBox, newLeftBBox)) {
                currentReferences[leftCount++] = objlist->size() +
                  clippedReferences.size();
                clippedReferences.push_back({reference, newLeftBBox});
              } else
                numPrunedTriangles++;
              if (OverlapS(rightBBox, newRightBBox)) {
                rightReferences.push_back(objlist->size() +
                  clippedReferences.size());
                clippedReferences.push_back({reference, newRightBBox});
              } else
                numPrunedTriangles++;
            } else {
              currentReferences[leftCount++] = objlist->size() +
                clippedReferences.size();
              clippedReferences.push_back({reference, newLeftBBox});
              rightReferences.push_back(objlist->size() +
                clippedReferences.size());
              clippedReferences.push_back({reference, newRightBBox});
            }
 
          // No other case is possible
          } else
            abort();
        }
 
      } else if (overlapsLeft) {
        currentReferences[leftCount++] = reference;
      } else if (overlapsRight) {
        rightReferences.push_back(reference);
      }
    }
    currentReferences.resize(leftCount);
 
  } else {
    unsigned leftCount = 0;
    for (unsigned i = 0; i < currentReferences.size(); i++) {
      const unsigned reference = currentReferences[i];
      const auto &objectBBox = GetObjectBBox(reference);
      if (OverlapS(leftBBox, objectBBox))
        currentReferences[leftCount++] = reference;
      if (OverlapS(rightBBox, objectBBox))
        rightReferences.push_back(reference);
    }
    currentReferences.resize(leftCount);
  }
}
 
glm::vec3
CKDTree::SplitPlaneIntersection(const unsigned axis, const float split,
  const glm::vec3 &a, const glm::vec3 &b)
{
  const glm::vec3 delta = b - a;
  const float dt = (split - a[axis]) / delta[axis];
  const glm::vec3 intersection = a + dt * delta;
  return intersection;
}
 
void
CKDTree::CreateChildren(const unsigned index, vector<unsigned> &leftReferences,
  vector<unsigned> &rightReferences, const SBBox &leftBBox,
  const SBBox &rightBBox)
{
  const int depthBackup = currentDepth + 1;
 
  // Left child
  currentDepth = depthBackup;
  BuildRecursively(leftReferences, leftBBox);
 
  // Right child
  nodes[index].SetReference(nodes.size());
  currentDepth = depthBackup;
  BuildRecursively(rightReferences, rightBBox);
}
 
const CObject3D*
CKDTree::FindNearestI(CRay &ray, CHitPointInfo &info)
{
  float tmin = 0.0f, tmax = CLimits::Infinity;
  info.SetMaxT(Min(info.GetMaxT(), tmax));
  const float saveMaxT = info.GetMaxT();
 
  if (!bbox.RayIntersect(ray, tmin, tmax))
    return nullptr;
 
  auto rayDirection = ray.GetDir();
  for (unsigned i = 0; i < 3; i++)
    if (rayDirection[i] == 0.0f)
      rayDirection[i] = CLimits::Small;
  const auto inverseDirection = 1.0f / rayDirection;
 
  // Recursive traversal
  stack.clear();
  stack.emplace_back(0, tmin, tmax);
  unsigned index;
  while (!stack.empty()) {
    numTraversalSteps++;
    index = stack.back().index;
    tmin = stack.back().tmin;
    tmax = stack.back().tmax;
    stack.pop_back();
 
    if (tmin >= info.GetMaxT())
      continue;
 
    // Internal node
    while (!nodes[index].IsLeaf()) {
      const unsigned axis = nodes[index].GetAxis();
      const float t = inverseDirection[axis] *
        (nodes[index].split - ray.GetLoc()[axis]);
 
      unsigned far = index + 1;
      unsigned near = nodes[index].GetReference();
      if (ray.GetDir()[axis] > 0.0f)
        swap(near, far);
 
      if (t >= tmax)
        index = near;
      else if (t <= tmin)
        index = far;
      else {
        stack.emplace_back(far, t, tmax);
        index = near;
        tmax = t;
      }
    }
 
    // Leaf node
    const auto reference = nodes[index].GetReference();
    if (reference) {
      const auto count = references[reference];
      for (unsigned i = 1; i <= count; i++) {
        numIncidenceTests++;
        const auto object = (*objlist)[GetObjectReference(
          references[reference + i])];
        if (object->NearestInt(ray, info))
          if (info.GetT() >= tmin && info.GetT() <= tmax)
            info.SetMaxT(info.GetT());
      }
    }
  }
 
  info.SetT(info.GetMaxT());
  info.SetMaxT(saveMaxT);
 
  return info.GetObject3D();
}
