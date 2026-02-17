//-----------------------------------------------------------------------------
/**
* \file   select.cpp
* \author Jakub Profota
* \brief  Scene and node selects.
*
* This file contains scene and node selection methods.
*/
//-----------------------------------------------------------------------------


#include "pgr.h"

#include "../global.h"
#include "../scene/node.h"
#include "../scene/scene.h"


enum ModifierEnum {
    SHIFT,
    CONTROL,
    NONE
};


Node *Scene::select(unsigned char index, int action) {
    return table->select(index, action);
}


Node *Table::select(unsigned char index, int action) {
    if (index == this->index) {
        return this;
    }

    // Saucers.
    for (std::vector<Saucer *>::iterator i = saucers.begin();
        i != saucers.end(); ++i) {
        Node *node = (*i)->select(index, action);
        if (node != nullptr && node->index == index) {
            return node;
        }
    }

    // Butterflies.
    for (std::vector<Butterfly *>::iterator i = butterflies.begin();
        i != butterflies.end(); ++i) {
        Node *node = (*i)->select(index, action);
        if (node != nullptr && node->index == index) {
            return node;
        }
    }

    // Lamp.
    if (this->lamp) {
        return this->lamp->select(index, action);
    }
}


Node *Saucer::select(unsigned char index, int action) {
    if (index == this->index) {
        if (action == SHIFT) {
        } else if (action == CONTROL) {
        } else {
            if (name == "white") {
                name = "gray";
            } else if (name == "gray") {
                name = "black";
            } else if (name == "black") {
                name = "red";
            } else if (name == "red") {
                name = "green";
            } else if (name == "green") {
                name = "blue";
            } else if (name == "blue") {
                name = "yellow";
            } else if (name == "yellow") {
                name = "cyan";
            } else if (name == "cyan") {
                name = "magenta";
            } else if (name == "magenta") {
                name = "white";
            }
        }

        return this;
    }

    // Vase.
    if (this->vase) {
        return this->vase->select(index, action);
    }
}


Node *Vase::select(unsigned char index, int action) {
    if (index == this->index) {
        if (action == SHIFT) {
        } else if (action == CONTROL) {
        } else {
            if (name == "bronze") {
                name = "egypt";
            } else if (name == "egypt") {
                name = "japan";
            } else if (name == "japan") {
                name = "porcelain";
            } else if (name == "porcelain") {
                name = "terracota";
            } else if (name == "terracota") {
                name = "bronze";
            }
        }

        return this;
    }

    // Flowers.
    for (std::vector<Flower *>::iterator i = flowers.begin();
        i != flowers.end(); ++i) {
        Node *node = (*i)->select(index, action);
        if (node != nullptr && node->index == index) {
            return node;
        }
    }
}


Node *Flower::select(unsigned char index, int action) {
    if (index == this->index) {
        if (action == SHIFT) {
        } else if (action == CONTROL) {
        } else {
            if (name == "bergenia") {
                name = "crocus";
            } else if (name == "crocus") {
                name = "geranium";
            } else if (name == "geranium") {
                name = "hibiscous";
            } else if (name == "hibiscous") {
                name = "lilies";
            } else if (name == "lilies") {
                name = "rose";
            } else if (name == "rose") {
                name = "sunflower";
            } else if (name == "sunflower") {
                name = "bergenia";
            }
        }

        return this;
    } else {
        return nullptr;
    }
}


Node *Butterfly::select(unsigned char index, int action) {
    if (index == this->index) {
        return this;
    } else {
        return nullptr;
    }
}


Node *Lamp::select(unsigned char index, int action) {
    if (index == this->index) {
        return this;
    } else {
        return nullptr;
    }
}


//-----------------------------------------------------------------------------
