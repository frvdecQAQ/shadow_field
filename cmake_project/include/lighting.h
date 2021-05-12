#ifndef LIGHT_H_
#define LIGHT_H_

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include "sampler.h"
#include "shorder.hpp"
#include "shRotateMatrix.h"

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
    void querySRF(glm::vec3 p, glm::vec3* coef);
    void rotate(const glm::mat4& m);

    int band() { return _band; }

    std::string save_path;
    std::string cube_map_path;
    int light_type;

    std::vector<glm::vec3> _coeffs;
    Eigen::VectorXf _Vcoeffs[3];

    std::vector<float> _vertices;
    std::vector<float> _texcoords;
    std::vector<float> _normals;
    std::vector<float> _indices;

    float _cx, _cy, _cz, _r;
    float init_x, init_y, init_z;
    glm::mat4 rotate_mat;
    glm::mat4 rotate_mat_inv;

    const int sphereNumber = 64;
    const int shadowSampleNumber = 128 * 128;
    const float rStep = 0.15f;
    const float rStart = 0.2f;

    glm::vec3* shadow_field = nullptr;
    Sampler point_sample;

private:
    std::string _path;
    std::string _filename;
    // The band of SH basis function.
    int _band;

    int _width;
    int _height;
    float* _data = nullptr;

    glm::vec3 probeColor(glm::vec3 dir);
    unsigned index_from_str(const std::string& str);
};

#endif
