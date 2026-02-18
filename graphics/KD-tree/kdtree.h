/// \file kdtree.h
/// \author Jakub Profota
/// \date 2024
///
/// \brief File contains implementation of clipping pruning kd-tree utilizing
/// binning SAH strategy.
 
#ifndef __KDTREE_H__
#define __KDTREE_H__
 
/// \class CKDTree
/// \brief Clipping pruning kd-tree utilizing binning SAH strategy.
///
/// CKDTree class implements kd-tree spatial data structure with clipping and
/// pruning capabilities. It uses either spatial median splitting or computation
/// of surface area heuristic (SAH) cost using binning strategy.
///
/// \section Binning
/// The user can choose between the spatial median and binning strategy
/// in the environment configuration file. The number of bins can also be set.
/// Additionaly, the user can choose whether the binning SAH computation should
/// consider all three dimensions, or only the dimension with the largest
/// extent.
///
/// \section Clipping
/// When clipping is enabled, the kd-tree splits the bounding box of a triangle
/// straddling the splitting plane into two. When clipping is off, the same
/// bounding box of the triangle is passed to both children.
///
/// \section Pruning
/// Creation of the leaf node commences when the number of references is less
/// than or the current recursion depth is greater than some user-defined value.
/// When the leaf is being created and pruning is enabled, the kd-tree checks
/// if each referenced triangle lies within the bounding box of the leaf node
/// using the separating axis theorem. Furthemore, if clipping is enabled,
/// the kd-tree discards those bounding boxes constructed when straddling
/// the splitting plane, which do not overlap the bounding box of the current
/// node.
///
/// \see CASDS_BB, SBBox, CObjectList, CRay, CHitPointInfo
class CKDTree:
  public CASDS_BB
{
protected:
 
  /// \struct SKDStackEntry
  /// \brief Stack entry for the kd-tree traversal.
  ///
  /// SKDStackEntry structure represents an entry in the stack used
  /// for the traversal of the kd-tree during ray tracing. It contains
  /// the index of the corresponding node and entry and exit distances.
  struct SKDStackEntry {
    /// Index of the node.
    const unsigned index;
    /// Entry distance.
    const float tmin;
    /// Exit distance.
    const float tmax;
 
    /// \brief Constructor.
    ///
    /// \param[in] index Index of the node.
    /// \param[in] tmin Entry distance.
    /// \param[in] tmax Exit distance.
    SKDStackEntry(const unsigned index, const float tmin, const float tmax):
      index(index), tmin(tmin), tmax(tmax) {}
  };
 
  /// \struct SKDNode
  /// \brief Node of the kd-tree.
  ///
  /// SKDNode structure represents a node of the kd-tree. It contains
  /// the position of the splitting plane and an index with additional
  /// information about the node. If the node is an internal node, the index
  /// holds the position of the right child in the nodes array. If the node
  /// is a leaf, the index points to the beginning of contained triangles.
  struct SKDNode {
    /// Position of the splitting plane.
    float split;
    /// Index to either the nodes or triangle references array. The two least
    /// significant bits are used to determine whether the node is a leaf and,
    /// if not, the axis of the splitting plane.
    unsigned index;
 
    /// \brief Marks the node as a leaf node.
    ///
    /// Sets the two least significant bits to 1.
    inline void MarkAsLeaf() { index = 0b11u; }
 
    /// \brief Checks whether the node is a leaf node.
    ///
    /// Checks whether the two least significant bits are set to 1.
    ///
    /// \return True if the node is a leaf node, false otherwise.
    inline bool IsLeaf() const { return (index & 0b11u) == 0b11u; }
 
    /// \brief Sets the axis of the splitting plane.
    ///
    /// The two least significant bits are reserved for the axis information.
    ///
    /// \param[in] axis Axis of the splitting plane.
    inline void SetAxis(const unsigned axis) { index = axis; }
 
    /// \brief Gets the axis of the splitting plane.
    ///
    /// The two least significant bits are reserved for the axis information.
    ///
    /// \return Axis of the splitting plane.
    inline unsigned GetAxis() const { return index & 0b11u; }
 
    /// \brief Sets the reference.
    ///
    /// Sets the reference either to the index of the right child or
    /// to the initial index of the contained triangles, depending on the node
    /// type. The index is shifted two bits to the left, since the two least
    /// significant bits are reserved for the axis information and the leaf
    /// flag.
    ///
    /// \param[in] reference Reference to the right child or the initial index
    /// of the contained triangles.
    inline void SetReference(const unsigned reference) {
      index |= reference << 2;
    }
 
    /// \brief Gets the reference.
    ///
    /// Gets the reference either to the index of the right child or
    /// to the initial index of the contained triangles, depending on the node
    /// type. The value is shifted two bits to the right, since the two least
    /// significant bits are reserved for the axis information and the leaf
    /// flag.
    ///
    /// \return Reference to the right child or the initial index of the
    /// contained triangles.
    inline unsigned GetReference() const { return index >> 2; }
  };
 
  /// Stack used for the traversal of the kd-tree during ray tracing.
  vector<SKDStackEntry> stack;
  /// Nodes of the kd-tree.
  vector<SKDNode> nodes;
 
  /// \brief References to the triangles.
  ///
  /// The element of this array where the index of a leaf node points to
  /// contains the number of referenced triangles in that node. Indices
  /// of the triangles, as stored in the object list, follow. If pruning
  /// is enabled, the index of an empty leaf node points to the very first index
  /// of this array, which is reserved and contains the number zero.
  vector<unsigned> references;
 
  /// \brief Clipped references to the triangles.
  ///
  /// If clipping is enabled and the kd-tree clips some triangles
  /// straddling the splitting plane, the newly constructred bounding boxes
  /// are stored in this array together with the original index
  /// of the triangle. If the index stored in a leaf node is larger than
  /// the size of the object list, than the index points to this array instead
  /// of the references array.
  vector<pair<unsigned, SBBox>> clippedReferences;
 
  /// \brief Array of the number of starting bounding boxes in each bin.
  ///
  /// If binning is enabled, the kd-tree uses this array to store
  /// the number of starting bounding boxes in each bin. The array is
  /// initialized with zeros and its size is equal to the number of bins.
  /// The array is used to compute the surface area heuristic (SAH) cost.
  vector<unsigned> binObjectStarts;
 
  /// \brief Array of the number of ending bounding boxes in each bin.
  ///
  /// If binning is enabled, the kd-tree uses this array to store
  /// the number of ending bounding boxes in each bin. The array is
  /// initialized with zeros and its size is equal to the number of bins.
  /// The array is used to compute the surface area heuristic (SAH) cost.
  vector<unsigned> binObjectEnds;
 
  /// \addtogroup KDTreeParameters
  /// \{
  /// Maximum depth of the kd-tree, or the depth in which the leaf node
  /// is created regardless of the number of references.
  int maxDepth;
  /// When the number of references in a node is less than this value,
  /// the node is marked as a leaf node.
  const unsigned maxLeafSize;
  /// If enabled, the kd-tree uses the binning strategy to compute the
  /// surface area heuristic (SAH) cost.
  const bool useBinning;
  /// If enabled, the kd-tree only considers the dimension with the largest
  /// extent when computing the surface area heuristic (SAH) cost.
  const bool useDrivingAxis;
  /// Number of bins used for the binning strategy.
  const int bins;
  /// Inverse of the number of bins, used to accelerate the computation.
  const float inverseBins;
  /// If enabled, the kd-tree clips the triangles straddling the splitting
  /// plane.
  const bool useClipping;
  /// If enabled, the kd-tree prunes the triangles that do not overlap
  /// the bounding box.
  const bool usePruning;
  /// \}
 
  /// Current depth of the kd-tree, used instead of passing the depth
  /// as a parameter to the recursive functions.
  int currentDepth;
 
  /// \addtogroup KDTreeStatistics
  /// \{
  /// Total number of incidence tests with triangles.
  int numIncidenceTests = 0;
  /// Total number of traversal steps through internal nodes.
  int numTraversalSteps = 0;
  /// Total number of internal nodes.
  int numInternalNodes = 0;
  /// Total number of leaf nodes.
  int numLeafNodes = 0;
  /// Ratio of the number of empty leaf nodes to the total number of leaf nodes.
  float ratioEmptyLeaves = 0.0f;
  /// Maximum depth of the tree.
  int maxLeafDepth = 0;
  /// Average depth of the tree.
  float avgLeafDepth = 0.0f;
  /// Total number of referenced triangles in leaf nodes.
  int numTriangles = 0;
  /// Average number of referenced triangles in a leaf node.
  float avgTriangles = 0.0f;
  /// Total number of clipped triangles.
  int numClippedTriangles = 0;
  /// Total number of pruned triangles.
  int numPrunedTriangles = 0;
  /// \}
 
public:
  /// Constructor.
  CKDTree();
 
  /// Constructor.
  ///
  /// \param[in] maxDepth Maximum depth of the kd-tree.
  /// \param[in] maxLeafSize Number of references to commence the leaf creation.
  /// \param[in] useBinning If enabled, the kd-tree uses the binning strategy.
  /// \param[in] useDrivingAxis If enabled, the kd-tree only considers the
  /// dimension with the largest extent.
  /// \param[in] bins Number of bins used for the binning strategy.
  /// \param[in] useClipping If enabled, the kd-tree clips the triangles
  /// straddling the splitting plane.
  /// \param[in] usePruning If enabled, the kd-tree prunes the triangles that
  /// do not overlap the bounding box.
  CKDTree(const int maxDepth, const int maxLeafSize, const bool useBinning,
    const bool useDrivingAxis, const int bins, const bool useClipping,
    const bool usePruning);
 
  /// Destructor.
  ~CKDTree();
 
  /// \brief Builds up the kd-tree.
  ///
  /// Recursively builds up the kd-tree spatial data structure. The function
  /// stores the object list and only works with the index references
  /// to the objects.
  ///
  /// \param[in] objectList Object list.
  virtual void BuildUp(const CObjectList &objectList);
 
  /// Removes the kd-tree.
  virtual void Remove();
 
  /// Returns the ID of the kd-tree.
  ///
  /// \return ID of the kd-tree.
  virtual EASDS_ID GetID() const { return ID_KDTree; }
 
  /// Prints out the ID of the kd-tree.
  ///
  /// \param[in] app Output stream.
  virtual void ProvideID(ostream &app);
 
  /// \brief Prints out the statistics of the kd-tree.
  void PrintStatistics();
 
  /// \brief Returns sign of the value.
  ///
  /// If the value is negative, the function returns -1. If the value is
  /// positive, 1 is returned. If the value is zero, the function returns 0.
  ///
  /// \param[in] value Value.
  inline int Sign(const float value) {
    return value <= 0.0f ? value == 0.0f ? 0 : -1 : 1;
  }
 
  /// Unionizes three bounding boxes.
  ///
  /// \param[in] a First bounding box.
  /// \param[in] b Second bounding box.
  /// \param[in] c Third bounding box.
  SBBox Union(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c);
 
  /// Unionizes four bounding boxes.
  ///
  /// \param[in] a First bounding box.
  /// \param[in] b Second bounding box.
  /// \param[in] c Third bounding box.
  /// \param[in] d Fourth bounding box.
  SBBox Union(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c,
    const glm::vec3 &d);
 
protected:
  /// \brief Returns the bounding box of the object.
  ///
  /// If the reference stored in a leaf node is smaller than the size
  /// of the object list, the function returns the bounding box of the object.
  /// If the reference stored is larger, then the reference points to a clipped
  /// triangle that straddled the splitting plane. In that case, the function
  /// returns the clipped bounding box.
  ///
  /// \param[in] reference Reference to the object.
  /// \return Bounding box of the object.
  inline SBBox GetObjectBBox(const unsigned reference) {
    if (reference >= objlist->size())
      return clippedReferences[reference - objlist->size()].second;
    else
      return (*objlist)[reference]->GetBox();
  }
 
  /// \brief Returns the reference to the object.
  ///
  /// If the reference stored in a leaf node is smaller than the size
  /// of the object list, the function returns the input reference.
  /// If the reference stored is larger, then the reference is computed
  /// by subtracting the size of the object list from the input reference.
  ///
  /// \param[in] reference Reference to the object.
  /// \return Reference to the object.
  inline unsigned GetObjectReference(const unsigned reference) {
    if (reference >= objlist->size())
      return clippedReferences[reference - objlist->size()].first;
    else
      return reference;
  }
 
  /// \brief Recursively build up the kd-tree.
  ///
  /// This recursive function builds up the kd-tree. First, it creates
  /// a new node. If the number of object references passed to the function
  /// is less than or the current depth is greater than a user-defined value,
  /// the function creates a leaf node and exits the recursion. Otherwise,
  /// the function finds a splitting plane according to the chosen strategy
  /// and splits the object references into two. The function then creates
  /// the left and right children and calls itself recursively for each child.
  ///
  /// \param[in] currentReferences Array of object references.
  /// \param[in] currentBBox Bounding box of the current node.
  void BuildRecursively(vector<unsigned> ¤tReferences,
    const SBBox ¤tBBox);
 
  /// \brief Creates a leaf node.
  ///
  /// Creates a leaf node by first storing the number of references objects
  /// in the references array, followed by the indices of the objects.
  /// If pruning is enabled, the function checks if the bounding box of each
  /// object overlaps the bounding box of the leaf node using the separating
  /// axis theorem, possibly discarding them. If the node is empty, the index
  /// points to the first element of the references array, which is reserved
  /// and contains the number zero.
  ///
  /// \param[in] index Index of the node.
  /// \param[in] currentReferences Array of object references.
  /// \param[in] currentBBox Bounding box of the current node.
  void CreateLeaf(const unsigned index,
    const vector<unsigned> ¤tReferences, const SBBox ¤tBBox);
 
  /// \brief Finds the splitting plane using the spatial median strategy.
  ///
  /// Finds the splitting plane by computing the spatial median
  /// of the dimension with the largest extent of the bounding box and stores
  /// the axis in the metadata part of the index of the node.
  ///
  /// \param[in] index Index of the node.
  /// \param[in] currentBBox Bounding box of the current node.
  void SpatialMedianSplit(const unsigned index, const SBBox ¤tBBox);
 
  /// \brief Finds the splitting plane using the binning strategy.
  ///
  /// Finds the splitting plane by computing the surface area heuristic (SAH)
  /// cost using the binning strategy. The function iterates over all three
  /// dimensions and finds the best splitting plane with the lowest cost.
  /// If set by the user, the function can only consider the dimension with
  /// the largest extent. The axis is stored in the metadata part of the index
  /// of the node.
  ///
  /// \param[in] index Index of the node.
  /// \param[in] currentReferences Array of object references.
  /// \param[in] currentBBox Bounding box of the current node.
  /// \param[out] rightCount Number of references in the right child, used
  /// to reserve space for the right child in advance.
  void SAHBinningSplit(const unsigned index,
    const vector<unsigned> ¤tReferences, const SBBox ¤tBBox,
    unsigned &rightCount);
 
  /// \brief Bins the object references to the bins.
  ///
  /// This function is used while searching for the best splitting plane
  /// using the binning strategy. It bins the object references to the bins
  /// and stores the number of starting and ending bounding boxes in each bin.
  ///
  /// \param[in] currentReferences Array of object references.
  /// \param[in] axis Axis of the splitting plane.
  /// \param[in] currentBBox Bounding box of the current node.
  void BinReferences(const vector<unsigned> ¤tReferences,
    const unsigned axis, const SBBox ¤tBBox);
 
  /// \brief Finds the splitting plane with the best cost after sorting
  /// the references to bins.
  ///
  /// This function is used while searching for the best splitting plane
  /// using the binning strategy. After the references are sorted to bins,
  /// the function finds the best splitting plane with the lowest cost
  /// by iterating over the bins and computing the cost for each splittting
  /// plane candidate. The best cost and the splitting plane are stored
  /// in the node.
  ///
  /// \param[in] index Index of the node.
  /// \param[in] currentReferences Array of object references.
  /// \param[in] axis Axis of the splitting plane.
  /// \param[in] currentBBox Bounding box of the current node.
  /// \param[in,out] leftBBox Bounding box of the left child.
  /// \param[in,out] rightBBox Bounding box of the right child.
  /// \param[out] rightCount Number of references in the right child, used
  /// to reserve space for the right child in advance.
  /// \param[out] bestCost Best cost of the splitting plane.
  void FindBinningCost(const unsigned index,
    const vector<unsigned> ¤tReferences, const unsigned axis,
    const SBBox ¤tBBox, SBBox &leftBBox, SBBox &rightBBox,
    unsigned &rightCount, float &bestCost);
 
  /// \brief Partitions the object references.
  ///
  /// Sorts the object references to the left and right children according
  /// to their position relative to the splitting plane. If clipping is
  /// disabled, the function checks whether the bounding box of the triangle
  /// overlaps the bounding box of the left or right child. If it does,
  /// the reference is passed to the recursive call for the corresponding child.
  /// If clipping is enabled, the function clips the bounding box
  /// of the triangle if it straddles the splitting plane, resulting in each
  /// bounding box being passed to only one child during the recursion. There
  /// are five possible cases for the bounding box clipping, the reader is
  /// advised to look at the implementation for more details.
  ///
  /// \param[in,out] currentReferences Array of input object references, which
  /// is also used as an array of the left child references.
  /// \param[out] rightReferences Array of object references for the right
  /// child.
  /// \param[in] leftBBox Bounding box of the left child.
  /// \param[in] rightBBox Bounding box of the right child.
  /// \param[in] axis Axis of the splitting plane.
  /// \param[in] split Position of the splitting plane.
  ///
  /// \see kdtree.cpp
  void PartitionReferences(vector<unsigned> ¤tReferences,
    vector<unsigned> &rightReferences, const SBBox &leftBBox,
    const SBBox &rightBBox, const unsigned axis, const float split);
 
  /// \brief Finds the intersection of the splitting plane and the line segment.
  ///
  /// This function is used while clipping the bounding boxes of the triangles
  /// straddling the splitting plane.
  ///
  /// \param[in] axis Axis of the splitting plane.
  /// \param[in] split Position of the splitting plane.
  /// \param[in] a First point of the line segment.
  /// \param[in] b Second point of the line segment.
  /// \return Intersection point of the splitting plane and the line segment.
  glm::vec3 SplitPlaneIntersection(const unsigned axis, const float split,
    const glm::vec3 &a, const glm::vec3 &b);
 
  /// \brief Creates the children of the node.
  ///
  /// Recursively builds left and right children of the node.
  ///
  /// \param[in] index Index of the node.
  /// \param[in] leftReferences Array of object references for the left child.
  /// \param[in] rightReferences Array of object references for the right child.
  /// \param[in] leftBBox Bounding box of the left child.
  /// \param[in] rightBBox Bounding box of the right child.
  void CreateChildren(const unsigned index, vector<unsigned> &leftReferences,
    vector<unsigned> &rightReferences, const SBBox &leftBBox,
    const SBBox &rightBBox);
 
  /// \brief Finds the nearest intersection of the ray with the kd-tree.
  ///
  /// Traverses the kd-tree using the stack and finds the intersection
  /// of the ray with the nearest object, if there is any.
  ///
  /// \param[in] ray Ray.
  /// \param[in,out] info Information about the hit point, if there is any.
  /// Contains the intersected object, the intersection point, the hit
  /// distance, and other information.
  /// \return Nearest object intersected by the ray, or nullptr if there is
  /// no intersection.
  ///
  /// \see CObject3D, CRay, CHitPointInfo
  virtual const CObject3D* FindNearestI(CRay &ray, CHitPointInfo &info);
};

#endif // __KDTREE_H__