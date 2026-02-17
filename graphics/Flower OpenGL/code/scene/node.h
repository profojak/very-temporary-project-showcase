//-----------------------------------------------------------------------------
/**
 * \file   node.h
 * \author Jakub Profota
 * \brief  Scene nodes.
 *
 * This file contains scene hierarchy nodes.
 */
//-----------------------------------------------------------------------------


#ifndef NODE_H
#define NODE_H


#include "pgr.h"

#include <string>
#include <vector>


class Table;
class Saucer;
class Vase;
class Flower;
class Butterfly;
class Bug;
class Spider;
class Lamp;


//-----------------------------------------------------------------------------


class Node {
public:
    /// Constructor.
    /**
     * \param[in] name Node name.
     * \param[in] index Stencil buffer index.
     */
    Node(const std::string name, unsigned char index);


    /// Destructor.
    virtual ~Node(void);


    /// Add child node.
    /**
     * \param[in] node Child node to add.
     */
    virtual void add_child(Node *node);


    /// Remove child node.
    /**
     * \param[in] node Child node to remove.
     */
    virtual void remove_child(Node *node);


    /// Set parent node.
    /**
     * \param[in] node Node to set as the parent node.
     */
    virtual void set_parent(Node *node);


    /// Outputs the node and children to standard output.
    /**
     * \param[in] indent Indentation level.
     */
    virtual void print(unsigned indent = 0);


    /// Updates the node.
    /**
     * \param[in] global_position Global position.
     * \param[in] global_rotation Global rotation.
     */
    virtual void update(glm::vec2 &global_position, glm::vec2 &global_rotation);



    /// Draws the node on the screen.
    /**
     * \param[in] P Perspective projection matrix.
     * \param[in] V View transformation matrix.
     */
    virtual void draw(glm::mat4 &P, glm::mat4 &V);


    /// Mouse selection.
    Node *select(unsigned char index, int action);


    std::string name; ///< Node name.
    glm::vec2 local_position;  ///< Position on the table.
    glm::vec2 local_rotation;  ///< Rotation on the table.
    glm::vec2 global_position; ///< Position in the world.
    glm::vec2 global_rotation; ///< Rotation in the world.
    unsigned char index;       ///< Stencil buffer index.
};


//-----------------------------------------------------------------------------


/// Class that holds information about table node.
class Table:public Node {
public:
    /// Constructor.
    /**
     * \param[in] name   Node name.
     * \param[in] length Length of the table.
     * \param[in] width  Width of the table.
     * \param[in] index Stencil buffer index.
     */
    Table(const std::string name, float length, float width,
        unsigned char index);


    /// Destructor.
    ~Table(void);


    /// Add child node.
    /**
     * \param[in] node Child node to add.
     */
    void add_child(Node *node) override;


    /// Remove child node.
    /**
     * \param[in] node Child node to remove.
     */
    void remove_child(Node *node) override;


    /// Outputs the node and children to standard output.
    /**
     * \param[in] indent Indentation level.
     */
    void print(unsigned indent = 0);


    /// Updates the node.
    /**
     * \param[in] global_position Global position.
     * \param[in] global_rotation Global rotation.
     */
    void update(glm::vec2 &global_position, glm::vec2 &global_rotation) override;


    /// Draws the node on the screen.
    /**
     * \param[in] P Perspective projection matrix.
     * \param[in] V View transformation matrix.
     */
    void draw(glm::mat4 &P, glm::mat4 &V) override;


    /// Mouse selection.
    Node *select(unsigned char index, int action);


    glm::vec2 size; ///< Size of the table.


private:
    std::vector<Saucer *> saucers;        ///< Vector of child saucers.
    std::vector<Butterfly *> butterflies; ///< Vector of child butterflies.
    Bug *bug;                             ///< Child bug.
    Spider *spider;                       ///< Child spider.
    Lamp *lamp;                           ///< Child lamp.
};


//-----------------------------------------------------------------------------


/// Class that holds information about saucer node.
class Saucer:public Node {
public:
    /// Constructor.
    /**
     * \param[in] name   Node name.
     * \param[in] parent Parent node.
     * \param[in] x      X coordinate on the table.
     * \param[in] y      Y coordinate on the table.
     * \param[in] index Stencil buffer index.
     */
    Saucer(const std::string name, Table *parent, float x, float y,
        unsigned char index);


    /// Destructor.
    ~Saucer(void);


    /// Add child node.
    /**
     * \param[in] node Child node to add.
     */
    void add_child(Node *node) override;


    /// Remove child node.
    /**
     * \param[in] node Child node to remove.
     */
    void remove_child(Node *node) override;


    /// Set parent node.
    /**
     * \param[in] node Node to set as the parent node.
     */
    void set_parent(Node *node) override;


    /// Outputs the scene node and children to standard output.
    /**
     * \param[in] indent Indentation level.
     */
    void print(unsigned indent = 0);


    /// Updates the node.
    /**
     * \param[in] global_position Global position.
     * \param[in] global_rotation Global rotation.
     */
    void update(glm::vec2 &global_position, glm::vec2 &global_rotation) override;


    /// Draws the node on the screen.
    /**
     * \param[in] P Perspective projection matrix.
     * \param[in] V View transformation matrix.
     */
    void draw(glm::mat4 &P, glm::mat4 &V) override;


    /// Mouse selection.
    Node *select(unsigned char index, int action);


private:
    Vase *vase;   ///< Child vase.
    Table *table; ///< Parent table.
};


//-----------------------------------------------------------------------------


// Class that holds information about vase node.
class Vase:public Node {
public:
    /// Constructor.
    /**
     * \param[in] name   Node name.
     * \param[in] parent Parent node.
     * \param[in] index Stencil buffer index.
     */
    Vase(const std::string name, Saucer *parent,
        unsigned char index);


    /// Destructor.
    ~Vase(void);


    /// Add child node.
    /**
     * \param[in] node Child node to add.
     */
    void add_child(Node *node) override;


    /// Remove child node.
    /**
     * \param[in] node Child node to remove.
     */
    void remove_child(Node *node) override;


    /// Set parent node.
    /**
     * \param[in] node Node to set as the parent node.
     */
    void set_parent(Node *node) override;


    /// Outputs the scene node and children to standard output.
    /**
     * \param[in] indent Indentation level.
     */
    void print(unsigned indent = 0);


    /// Updates the node.
    /**
     * \param[in] global_position Global position.
     * \param[in] global_rotation Global rotation.
     */
    void update(glm::vec2 &global_position, glm::vec2 &global_rotation) override;


    /// Draws the node on the screen.
    /**
     * \param[in] P Perspective projection matrix.
     * \param[in] V View transformation matrix.
     */
    void draw(glm::mat4 &P, glm::mat4 &V) override;


    /// Mouse selection.
    Node *select(unsigned char index, int action);


private:
    std::vector<Flower *> flowers; ///< Vector of child flowers.
    Saucer *saucer;                ///< Parent saucer.
};


//-----------------------------------------------------------------------------


// Class that holds information about flower node.
class Flower:public Node {
public:
    /// Constructor.
    /**
     * \param[in] name   Node name.
     * \param[in] parent Parent node.
     * \param[in] index Stencil buffer index.
     */
    Flower(const std::string name, Vase *parent,
        unsigned char index);


    /// Destructor.
    ~Flower(void);


    /// Set parent node.
    /**
     * \param[in] node Node to set as the parent node.
     */
    void set_parent(Node *node) override;


    /// Updates the node.
    /**
     * \param[in] global_position Global position.
     * \param[in] global_rotation Global rotation.
     */
    void update(glm::vec2 &global_position, glm::vec2 &global_rotation) override;


    /// Draws the node on the screen.
    /**
     * \param[in] P Perspective projection matrix.
     * \param[in] V View transformation matrix.
     */
    void draw(glm::mat4 &P, glm::mat4 &V) override;


    /// Mouse selection.
    Node *select(unsigned char index, int action);


private:
    Vase *vase; ///< Parent vase.
};


//-----------------------------------------------------------------------------


/// Class that holds information about butterfly node.
class Butterfly:public Node {
public:
    /// Constructor.
    /**
     * \param[in] name   Node name.
     * \param[in] parent Parent node.
     * \param[in] x      X position.
     * \param[in] y      Y position.
     * \param[in] index Stencil buffer index.
     */
    Butterfly(const std::string name, Table *parent, float x, float y,
        unsigned char index);


    /// Destructor.
    ~Butterfly(void);


    /// Set parent node.
    /**
     * \param[in] node Node to set as the parent node.
     */
    void set_parent(Node *node) override;


    /// Outputs the scene node to standard output.
    /**
     * \param[in] indent Indentation level.
     */
    void print(unsigned indent = 0);


    /// Updates the node.
    /**
     * \param[in] global_position Global position.
     * \param[in] global_rotation Global rotation.
     */
    void update(glm::vec2 &global_position, glm::vec2 &global_rotation) override;


    /// Draws the node on the screen.
    /**
     * \param[in] P Perspective projection matrix.
     * \param[in] V View transformation matrix.
     */
    void draw(glm::mat4 &P, glm::mat4 &V) override;


    /// Mouse selection.
    Node *select(unsigned char index, int action);


private:
    Table *table; ///< Parent table.
};


//-----------------------------------------------------------------------------


/// Class that holds information about bug node.
class Bug:public Node {
public:
    /// Constructor.
    /**
    * \param[in] name   Node name.
    * \param[in] parent Parent node.
    */
    Bug(const std::string name, Table *parent);


    /// Destructor.
    ~Bug(void);


    /// Set parent node.
    /**
    * \param[in] node Node to set as the parent node.
    */
    void set_parent(Node *node) override;


    /// Draws the node on the screen.
    /**
    * \param[in] P Perspective projection matrix.
    * \param[in] V View transformation matrix.
    */
    void draw(glm::mat4 &P, glm::mat4 &V) override;


private:
    Table *table; ///< Parent table.
};


//-----------------------------------------------------------------------------


/// Class that holds information about spider node.
class Spider:public Node {
public:
    /// Constructor.
    /**
    * \param[in] name   Node name.
    * \param[in] parent Parent node.
    */
    Spider(const std::string name, Table *parent);


    /// Destructor.
    ~Spider(void);


    /// Set parent node.
    /**
    * \param[in] node Node to set as the parent node.
    */
    void set_parent(Node *node) override;


    /// Draws the node on the screen.
    /**
    * \param[in] P Perspective projection matrix.
    * \param[in] V View transformation matrix.
    */
    void draw(glm::mat4 &P, glm::mat4 &V) override;


private:
    Table *table; ///< Parent table.
};


//-----------------------------------------------------------------------------


/// Class that holds information about lamp node.
class Lamp:public Node {
public:
    /// Constructor.
    /**
     * \param[in] name   Node name.
     * \param[in] parent Parent node.
     * \param[in] x      Position on the edge of the table.
     * \param[in] index Stencil buffer index.
     */
    Lamp(const std::string name, Table *parent, float x,
        unsigned char index);


    /// Destructor.
    ~Lamp(void);


    /// Set parent node.
    /**
     * \param[in] node Node to set as the parent node.
     */
    void set_parent(Node *node) override;


    /// Updates the node.
    /**
     * \param[in] global_position Global position.
     * \param[in] global_rotation Global rotation.
     */
    void update(glm::vec2 &global_position, glm::vec2 &global_rotation) override;


    /// Draws the node on the screen.
    /**
     * \param[in] P Perspective projection matrix.
     * \param[in] V View transformation matrix.
     */
    void draw(glm::mat4 &P, glm::mat4 &V) override;


    /// Mouse selection.
    Node *select(unsigned char index, int action);


private:
    Table *table; ///< Parent table.
};


//-----------------------------------------------------------------------------


#endif //!NODE_H
