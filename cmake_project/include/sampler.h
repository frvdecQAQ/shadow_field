#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <vector>
#include <glm/glm.hpp>

class Sample
{
public:
    glm::vec3 _cartesCoord;
    glm::vec2 _sphericalCoord; // theta, phi
    float* _SHvalue;

    Sample(glm::vec3 car_in, glm::vec2 sph_in): _cartesCoord(car_in), _sphericalCoord(sph_in)
    {
    }
};

class Sampler
{
public:
    // sqrt of sample number.
    Sampler();
    Sampler(unsigned int n, bool random_offset = true);
    // band means l.
    void computeSH(int band);

    std::vector<Sample> _samples;
};


#endif
