#ifndef LIGHT_H_
#define LIGHT_H_

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <glm/glm.hpp>

class Lighting
{
public:
    Lighting() = default;
    // Constructor for preprocessing.
    Lighting(std::string path, int band);
    //Lighting(int band, Eigen::VectorXf coeffs[3]);

    ~Lighting();

    void init();
    void process(int sampleNumber);
    void write2Diskbin();

    int band() { return _band; }

    std::string save_path;
    std::string cube_map_path;
    int light_type;

    std::vector<glm::vec3> _coeffs;
    Eigen::VectorXf _Vcoeffs[3];

private:
    std::string _path;
    std::string _filename;
    // The band of SH basis function.
    int _band;

    int _width;
    int _height;
    float* _data = nullptr;

    glm::vec3 probeColor(glm::vec3 dir);
};

#endif
