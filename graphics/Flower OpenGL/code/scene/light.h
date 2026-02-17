//-----------------------------------------------------------------------------
/**
* \file   light.h
* \author Jakub Profota
* \brief  Light.
*
* This file contains light class.
*/
//-----------------------------------------------------------------------------


#ifndef LIGHT_H
#define LIGHT_H


#include "pgr.h"


/// Class that holds light information.
class Light {
public:
    /// Constructor.
    Light(void):
        ambient(0.0f), diffuse(0.0f), specular(0.0f),
        position(0.0f), direction(0.0f),
        cone(0.0f), exponent(0.0f)
    {
    }


    /// Destructor.
    ~Light(void) {
    }


    glm::vec3 ambient;   // Ambient light component.
    glm::vec3 diffuse;   // Diffuse light component.
    glm::vec3 specular;  // Specular light component.
    glm::vec3 position;  // Light position.
    glm::vec3 direction; // Light direction.
    float cone;          // Cosine of the spotlight half angle.
    float exponent;      // Distribution of the energy within the light cone.
};


//-----------------------------------------------------------------------------


#endif //!LIGHT_H
