#ifndef DIFFUSEOBJECT_H_
#define DIFFUSEOBJECT_H_

#include <opencv2/opencv.hpp>
#include "sphericalHarmonics.h"
#include "bvhTree.h"
#include "sampler.h"
#include "object.h"
#include "shRotate.h"
#include "shRotateMatrix.h"
#include "shorder.hpp"

class DiffuseObject : public Object
{
public:
    DiffuseObject():Object()
    {
        _difforGeneral = false;
    }
    void project2SH(int mode, int band, int sampleNumber, int bounce) override;
    void write2Diskbin(std::string filename) override;
    void readFDiskbin(std::string filename) override;
    void transform(const glm::mat4& m);
    // For read.
    int debug_cnt = 0;
    std::vector<std::vector<glm::vec3>> _DTransferFunc;

private:
    void diffuseUnshadow(int size, int band2, Sampler* sampler, TransferType type, BVHTree* Inbvht = nullptr);
    void diffuseShadow(int size, int band2, Sampler* sampler, TransferType type, BVHTree* Inbvht = nullptr);
    void diffuseInterreflect(int size, int band2, Sampler* sampler, TransferType type, int bounce);
    float testCoef(float* coef, float theta, float phi);
    void testMap(float* coef, const std::string& path);
    // For write.
    std::vector<std::vector<glm::vec3>> _TransferFunc;
};

#endif
