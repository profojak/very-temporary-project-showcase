//-----------------------------------------------------------------------------
/**
 * \file   node.cpp
 * \author Jakub Profota
 * \brief  Scene nodes.
 *
 * This file contains scene hierarchy nodes.
 */
//-----------------------------------------------------------------------------


#include "node.h"

#include "../global.h"


/// Constructor.
Node::Node(const std::string name, unsigned char index):
    name(name),
    local_position(0.0f), local_rotation(0.0f),
    global_position(0.0f), global_rotation(0.0f),
    index(index)
{
}


/// Destructor.
Node::~Node(void) {
}


/// Add child node.
void Node::add_child(Node *node) {
    LOG_ERROR("Invalid function call!");
}


/// Remove child node.
void Node::remove_child(Node *node) {
    LOG_ERROR("Invalid function call!");
}


/// Set parent node.
void Node::set_parent(Node *node) {
    LOG_ERROR("Invalid function call!");
}


/// Outputs the node and children to standard output.
void Node::print(unsigned indent) {
    std::string ind(indent * 4, ' ');
    std::cout << ind << name << std::endl;
}


/// Updates the node.
void Node::update(glm::vec2 &global_position, glm::vec2 &global_rotation) {
    LOG_ERROR("Invalid function call!");
}


/// Draws the node on the screen.
void Node::draw(glm::mat4 &P, glm::mat4 &V) {
    LOG_ERROR("Invalid function call!");
}


//-----------------------------------------------------------------------------


/// Constructor.
Table::Table(const std::string name, float length, float width,
    unsigned char index):
    Node(name, index), lamp(nullptr)
{
    size.x = length;
    size.y = width;

    new Bug("bug", this);
    new Spider("spider", this);
}


/// Destructor.
Table::~Table(void) {
    // Saucers.
    for (std::vector<Saucer *>::iterator i = saucers.begin();
        i != saucers.end(); ++i) {
        delete(*i);
    }

    // Butterflies.
    for (std::vector<Butterfly *>::iterator i = butterflies.begin();
        i != butterflies.end(); ++i) {
        delete(*i);
    }

    // Bug.
    delete(this->bug);

    // Lamp.
    delete(this->lamp);
}


/// Add child node.
void Table::add_child(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Saucer *saucer = dynamic_cast<Saucer *>(node)) {
        saucer->set_parent(this);
        this->saucers.push_back(saucer);
    } else if (Butterfly *butterfly = dynamic_cast<Butterfly *>(node)) {
        butterfly->set_parent(this);
        this->butterflies.push_back(butterfly);
    } else if (Bug *bug = dynamic_cast<Bug *>(node)) {
        bug->set_parent(this);
        this->bug = bug;
    } else if (Spider *spider = dynamic_cast<Spider *>(node)) {
        spider->set_parent(this);
        this->spider = spider;
    } else if (Lamp *lamp = dynamic_cast<Lamp *>(node)) {
        if (this->lamp) {
            LOG_ERROR("Lamp is not null!");
            return;
        }
        lamp->set_parent(this);
        this->lamp = lamp;
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


/// Remove child node.
void Table::remove_child(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Saucer *saucer = dynamic_cast<Saucer *>(node)) {
        for (std::vector<Saucer *>::iterator i = saucers.begin();
            i != saucers.end(); ++i) {
            if (saucer == *i) {
                saucers.erase(i);
                break;
            }
        }
    } else if (Butterfly *butterfly = dynamic_cast<Butterfly *>(node)) {
        for (std::vector<Butterfly *>::iterator i = butterflies.begin();
            i != butterflies.end(); ++i) {
            if (butterfly == *i) {
                butterflies.erase(i);
                break;
            }
        }
    } else if (Lamp *lamp = dynamic_cast<Lamp *>(node)) {
        if (lamp == this->lamp) {
            delete(this->lamp);
            this->lamp = nullptr;
        }
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


/// Outputs the scene node and children to standard output.
void Table::print(unsigned indent) {
    std::string ind(indent * 4, ' ');
    std::cout << ind << name << " (" <<
        size.x << ", " << size.y << "):" << std::endl;

    // Saucers.
    for (std::vector<Saucer *>::iterator i = saucers.begin();
        i != saucers.end(); ++i) {
        (*i)->print(indent + 1);
    }

    // Butterflies.
    for (std::vector<Butterfly *>::iterator i = butterflies.begin();
        i != butterflies.end(); ++i) {
        (*i)->print(indent + 1);
    }

    // Lamp.
    if (this->lamp) {
        this->lamp->print(indent + 1);
    }
}


//-----------------------------------------------------------------------------


/// Constructor.
Saucer::Saucer(const std::string name, Table *parent, float x, float y,
    unsigned char index):
    Node(name, index), vase(nullptr), table(nullptr)
{
    parent->add_child(this);
    local_position.x = x;
    local_position.y = y;
}


/// Destructor.
Saucer::~Saucer(void) {
    delete(this->vase);
}


/// Add child node.
void Saucer::add_child(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Vase *vase = dynamic_cast<Vase *>(node)) {
        if (this->vase) {
            LOG_ERROR("Vase is not null!");
            return;
        }
        vase->set_parent(this);
        this->vase = vase;
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


/// Remove child node.
void Saucer::remove_child(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Vase *vase = dynamic_cast<Vase *>(node)) {
        if (vase == this->vase) {
            delete(this->vase);
            this->vase = nullptr;
        }
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


/// Set parent node.
void Saucer::set_parent(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Table *table = dynamic_cast<Table *>(node)) {
        if (this->table) {
            LOG_ERROR("Table is not null!");
        }
        this->table = table;
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


/// Outputs the scene node and children to standard output.
void Saucer::print(unsigned indent) {
    std::string ind(indent * 4, ' ');
    std::cout << ind << "" << name << " saucer (" <<
        local_position.x << ", " << local_position.y << "):" << std::endl;

    // Vase.
    if (this->vase) {
        this->vase->print(indent + 1);
    }
}


//-----------------------------------------------------------------------------


/// Constructor.
Vase::Vase(const std::string name, Saucer *parent, unsigned char index):
    Node(name, index), saucer(nullptr)
{
    parent->add_child(this);
}


/// Destructor.
Vase::~Vase(void) {
    // Flowers
    for (std::vector<Flower *>::iterator i = flowers.begin();
        i != flowers.end(); ++i) {
        delete(*i);
    }
}


/// Add child node.
void Vase::add_child(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Flower *flower = dynamic_cast<Flower *>(node)) {
        flower->set_parent(this);
        this->flowers.push_back(flower);
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


/// Remove child node.
void Vase::remove_child(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Flower *flower = dynamic_cast<Flower *>(node)) {
        for (std::vector<Flower *>::iterator i = flowers.begin();
            i != flowers.end(); ++i) {
            if (flower == *i) {
                flowers.erase(i);
                break;
            }
        }
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


/// Set parent node.
void Vase::set_parent(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Saucer *saucer = dynamic_cast<Saucer *>(node)) {
        if (this->saucer) {
            LOG_ERROR("Saucer is not null!");
            return;
        }
        this->saucer = saucer;
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


/// Outputs the scene node and children to standard output.
void Vase::print(unsigned indent) {
    std::string ind(indent * 4, ' ');
    std::cout << ind << name << " vase:" << std::endl;

    // Flowers.
    for (std::vector<Flower *>::iterator i = flowers.begin();
        i != flowers.end(); ++i) {
        (*i)->print(indent + 1);
    }
}


//-----------------------------------------------------------------------------


// Constructor.
Flower::Flower(const std::string name, Vase *parent, unsigned char index):
    Node(name, index), vase(nullptr)
{
    parent->add_child(this);
}


// Destructor.
Flower::~Flower(void) {
}


/// Set parent node.
void Flower::set_parent(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Vase *vase = dynamic_cast<Vase *>(node)) {
        if (this->vase) {
            LOG_ERROR("Vase is not null!");
        }
        this->vase = vase;
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


//-----------------------------------------------------------------------------


// Constructor.
Butterfly::Butterfly(const std::string name, Table *parent, float x, float y,
    unsigned char index):
    Node(name, index), table(nullptr)
{
    parent->add_child(this);

    local_position.x = x;
    local_position.y = y;
}


// Destructor.
Butterfly::~Butterfly(void) {
}


/// Set parent node.
void Butterfly::set_parent(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Table *table = dynamic_cast<Table *>(node)) {
        this->table = table;
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


/// Outputs the scene node to standard output.
void Butterfly::print(unsigned indent) {
    std::string ind(indent * 4, ' ');
    std::cout << ind << name << " butterfly" << std::endl;
}


//-----------------------------------------------------------------------------


// Constructor.
Bug::Bug(const std::string name, Table *parent):
    Node(name, 0), table(nullptr)
{
    parent->add_child(this);
}


// Destructor.
Bug::~Bug(void) {
}


/// Set parent node.
void Bug::set_parent(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Table *table = dynamic_cast<Table *>(node)) {
        this->table = table;
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


//-----------------------------------------------------------------------------


// Constructor.
Spider::Spider(const std::string name, Table *parent):
    Node(name, 0), table(nullptr)
{
    parent->add_child(this);
}


// Destructor.
Spider::~Spider(void) {
}


/// Set parent node.
void Spider::set_parent(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Table *table = dynamic_cast<Table *>(node)) {
        this->table = table;
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


//-----------------------------------------------------------------------------


/// Constructor.
Lamp::Lamp(const std::string name, Table *parent, float x, unsigned char index):
    Node(name, index), table(nullptr)
{
    parent->add_child(this);
}


/// Destructor.
Lamp::~Lamp(void) {
}


/// Set parent node.
void Lamp::set_parent(Node *node) {
    if (node == nullptr) {
        return;
    } else if (Table *table = dynamic_cast<Table *>(node)) {
        if (this->table) {
            LOG_ERROR("Table is not null!");
            return;
        }
        this->table = table;
    } else {
        LOG_ERROR("Invalid argument!");
    }
}


//-----------------------------------------------------------------------------
