#ifndef GENERALOBJECT_H_
#define GENERALOBJECT_H_

#include <vector>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include "bvhTree.h"
#include "object.h"
#include "sampler.h"

class GeneralObject : public Object
{
public:
    GeneralObject():Object()
    {
        _difforGeneral = true;
    }

    void project2SH(int mode, int band, int sampleNumber, int bounce) override;
    void write2Diskbin(std::string filename) override;
    void readFDiskbin(std::string filename) override;

    const float Kd = 0.1f;
    const float Ks = 1.0f;
    const int s = 10;
    std::vector<std::vector<float>> _TransferFunc;

private:
    void glossyUnshadow(int size, int band2, Sampler* sampler, TransferType type, BVHTree* Inbvht = nullptr);
    void glossyShadow(int size, int band2, Sampler* sampler, TransferType type, BVHTree* Inbvht = nullptr);
};

#endif
