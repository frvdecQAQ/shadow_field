#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "shproduct.h"
#include "shorder.hpp"
#include "sh.hpp"

template <int n>
matrix<n> operator* (const matrix<n>& a, const matrix<n>& b)
{
    matrix<n> c;
    memset(c.a, 0, sizeof c.a);
    for (int i=0; i<n; ++i)
        for (int j=0; j<n; ++j)
            for (int k=0; k<n; ++k)
                c.a[i][j] += a.a[i][k] * b.a[k][j];
    return c;
}


template<>
TensorEntry SH<3>::SparseGamma[] = {
#include "gamma/sparse3"
};
template<>
TensorEntry SH<4>::SparseGamma[] = {
#include "gamma/sparse4"
};
template<>
TensorEntry SH<5>::SparseGamma[] = {
#include "gamma/sparse5"
};
template<>
TensorEntry SH<6>::SparseGamma[] = {
#include "gamma/sparse6"
};
template<>
TensorEntry SH<7>::SparseGamma[] = {
#include "gamma/sparse7"
};
template<>
TensorEntry SH<8>::SparseGamma[] = {
#include "gamma/sparse8"
};
template<>
TensorEntry SH<9>::SparseGamma[] = {
#include "gamma/sparse9"
};
template<>
TensorEntry SH<10>::SparseGamma[] = {
#include "gamma/sparse10"
};
template<>
TensorEntry SH<11>::SparseGamma[] = {
#include "gamma/sparse11"
};
template<>
TensorEntry SH<12>::SparseGamma[] = {
#include "gamma/sparse12"
};
template<>
TensorEntry SH<13>::SparseGamma[] = {
#include "gamma/sparse13"
};
template<>
TensorEntry SH<15>::SparseGamma[] = {
#include "gamma/sparse15"
};
/*
template<>
TensorEntry SH<22>::SparseGamma[] = {
#include "gamma/sparse22"
};
*/

/*
template<>
TensorEntry SH<43>::SparseGamma[] = {
#include "gamma/src/sparse43"
};
*/

// projection of product of SH projected functions
SH<n> operator*(const SH<n>& a, const SH<n>& b)
{
    SH<n> c;
    for (auto e: SH<n>::SparseGamma)
        c.a[e.c] += e.val * a.a[e.a] * b.a[e.b];
    //for (auto it = std::begin(SH<n>::sparsegamma); it != std::end(SH<n>::sparsegamma); ++it)
	//c.a[it->c] += it->val * a.a[it->a] * b.a[it->b];
    return c;
}

// template <int n>
// SH<n> SH<n>::squared()
// {
//     SH<n> c;
//     for (auto e: SH<n>::SquareSparseGamma)
//         c.a[e.c] += e.val * a[e.a] * a[e.b];
//     return c;
// }


template <int n>
float dot(const SH<n>& a, const SH<n>& b)
{
    float t = 0;
    for (int i=0; i<n*n; ++i)
        t += a.a[i] * b.a[i];
    return t;
}


// projection of product as transformation lambda B: A*B
template <int n>
matrix<n*n> SH<n>::prodMatrix() const
{
    matrix<n*n> m;
    for (int i=0; i<n*n; ++i)
    for (int j=0; j<n*n; ++j)
    for (int k=0; k<n*n; ++k)
        m.a[i][j] += a[k] * SH<n>::Gamma[i][j][k];
    return m;
}

template <int n>
SH<n> operator*(const matrix<n*n>& a, const SH<n>& b)
{
    SH<n> c;
    for (int i=0; i<n*n; ++i)
        for (int j=0; j<n*n; ++j)
            c.a[i] += a.a[i][j] * b.a[j];
    return c;
}


std::vector<TensorEntry> deviceSparseGamma112;
int deviceSparseGamma112Size;
std::vector<TensorEntry> deviceSparseGamma213;
int deviceSparseGamma213Size;
std::vector<TensorEntry> deviceSparseGamma314;
int deviceSparseGamma314Size;
std::vector<TensorEntry> deviceSparseGamma411;
int deviceSparseGamma411Size;


std::vector<int> gammalist = {22,33,44,55,66,77};
constexpr int M = 4 * n - 3;

static std::vector<TensorEntry> readGamma(int n)
{
    int filen = -1;
    for (int a: gammalist)
        if (a>=n) {
            filen = a;
            break;
        }
    std::vector<TensorEntry> sparsegamma;
    std::ifstream sparsefile("./gamma_bin/" + std::to_string(filen) + ".bin");
    struct {int a,b,c;float k;} t;
    while (sparsefile.read((char*)(&t), 16)) {
        if (t.a<n*n && t.b<n*n && t.c<n*n)
            sparsegamma.push_back((TensorEntry){(short)t.a, (short)t.b, (short)t.c, t.k});
    }
    sparsefile.close();
    return sparsegamma;
}


std::vector<TensorEntry> filterGamma(std::vector<TensorEntry> v, int a, int b, int c)
{
        std::vector<TensorEntry> res;
        for (auto e: v) {
                if (e.a < a*a && e.b < b*b && e.c < c*c)
                res.push_back(e);
        }
        return res;
}

void cpu_initGamma()
{
      std::vector<TensorEntry> v;
      int size = 0;
      // load sparse gamma 3n-2
      v = readGamma(3*n-2);
      size = v.size();
      // gamma 1,1,2
      deviceSparseGamma112 = filterGamma(v, n, n, 2*n-1);
      deviceSparseGamma112Size = deviceSparseGamma112.size();
      // gamma 2,1,3
      deviceSparseGamma213 = filterGamma(v, 2*n-1, n, 3*n-2);
      deviceSparseGamma213Size = deviceSparseGamma213.size();

      v = readGamma(4*n-3);
      size = v.size();
      // gamma 3,1,1
      deviceSparseGamma314 = filterGamma(v, 3*n-2, n, 4*n-3);
      deviceSparseGamma314Size = deviceSparseGamma314.size();
      // gamma 4,1,1
      deviceSparseGamma411 = filterGamma(v, 4*n-3, n, n);
      deviceSparseGamma411Size = deviceSparseGamma411.size();
}

SH<n> precise(SH<n> sh1, SH<n> sh2, SH<n> sh3, SH<n> sh4, SH<n> sh5)
{
        SH<M> T1, T2, T3;
        SH<n> res;

        for(int i = 0; i < deviceSparseGamma112Size; ++i)
                T1.a[deviceSparseGamma112[i].c] += deviceSparseGamma112[i].val * sh1.a[deviceSparseGamma112[i].a] * sh2.a[deviceSparseGamma112[i].b];

        for(int i = 0; i < deviceSparseGamma213Size; ++i)
                T2.a[deviceSparseGamma213[i].c] += deviceSparseGamma213[i].val * T1.a[deviceSparseGamma213[i].a] * sh3.a[deviceSparseGamma213[i].b];

        for(int i = 0; i < deviceSparseGamma314Size; ++i)
                T3.a[deviceSparseGamma314[i].c] += deviceSparseGamma314[i].val * T2.a[deviceSparseGamma314[i].a] * sh4.a[deviceSparseGamma314[i].b];

        for(int i = 0; i < deviceSparseGamma411Size; ++i)
                res.a[deviceSparseGamma411[i].c] += deviceSparseGamma411[i].val * T3.a[deviceSparseGamma411[i].a] * sh5.a[deviceSparseGamma411[i].b];
    
        return res;
}