#ifndef SPHERICALHARMONICS_H_
#define SPHERICALHARMONICS_H_

#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "utils.h"

namespace SphericalH
{
    static std::vector<double>mem;
    static std::vector<bool>mem_vis;
    // Normalization constants.
    double static Kvalue(int l, int m)
    {
        if (m == 0)
        {
            return sqrt((2 * l + 1) / (4 * M_PI));
        }

        double up = (2 * l + 1) * factorial(l - abs(m));
        double down = (4 * M_PI) * factorial(l + abs(m));

        return sqrt(up / down);
    }

    // Value for Associated Legendre Polynomial.
    // Some information about Associated Legendre Polynomial:
    // https://en.wikipedia.org/wiki/Associated_Legendre_polynomials?oldformat=true
    double static evaluateLegendre(double x, int l, int m, bool check_mem = false)
    {
        if (check_mem) {
            int index = l * (l + 1) + m;
            if (mem_vis[index])return mem[index];
        }
        double result = 0.0;
        if (l == m)
        {
            result = minusOnePower(m) * doubleFactorial(2 * m - 1) * pow((1 - x * x), m / 2.0);
        }
        else if (l == m + 1)
        {
            result = x * (2 * m + 1) * evaluateLegendre(x, m, m, check_mem);
        }
        else
        {
            result = (x * (2 * l - 1) * evaluateLegendre(x, l - 1, m, check_mem) - (l + m - 1) * evaluateLegendre(x, l - 2, m, check_mem)) /
                (l - m);
        }
        if (check_mem) {
            int index = l * (l + 1) + m;
            mem_vis[index] = true;
            mem[index] = result;
        }
        return result;
    }

    // Value for Spherical Harmonic.
    double static SHvalue(double theta, double phi, int l, int m, bool check_mem = false)
    {
        double result = 0.0;
        if (m == 0)
        {
            result = Kvalue(l, 0) * evaluateLegendre(cos(theta), l, 0, check_mem);
        }
        else if (m > 0)
        {
            result = sqrt(2.0f) * Kvalue(l, m) * cos(m * phi) * evaluateLegendre(cos(theta), l, m, check_mem);
        }
        else
        {
            result = sqrt(2.0f) * Kvalue(l, m) * sin(-m * phi) * evaluateLegendre(cos(theta), l, -m, check_mem);
        }

        if (fabs(result) <= M_ZERO)
            result = 0.0;

        if (__isnan(result))
        {
            std::cout << "SPHERICAL HARMONIC NAN" << std::endl;
            std::cout << "theta: " << theta << " " << "phi: " << phi << std::endl;
        }
        return result;
    }

    void static SHvalueALL(int band, double theta, double phi, float* coef) {
        int band2 = band * band;
        mem.resize(band2, -1e10);
        mem_vis.resize(band2, false);
        for (int i = 0; i < band2; ++i)mem_vis[i] = false;
        for (int l = 0; l < band; ++l) {
            for (int m = -l; m <= l; ++m) {
                int index = l * (l + 1) + m;
                coef[index] = (float)SphericalH::SHvalue(theta, phi, l, m, true);
                if (__isnan(coef[index])){
                    std::cout << "NAN." << std::endl;
                    system("pause");
                }
            }
        }
    }

    void static testVisMap(int band, int n, const float* coef, const std::string store_path) {
        int band2 = band * band;
        cv::Mat gray(n, n, CV_32FC1);
        float* sh_value = new float[band2];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float x = (float)(i) / n;
                float y = (float)(j) / n;
                float theta = acos(1 - 2 * x);
                float phi = 2.0f * M_PI * y;
                for (int k = 0; k < band2; ++k)sh_value[k] = 0;
                SHvalueALL(band, theta, phi, sh_value);
                float pixel_value = 0;
                for (int k = 0; k < band2; ++k)pixel_value += coef[k] * sh_value[k];
                gray.at<float>(i, j) = pixel_value * 255.0;
                //std::cout << "theta = " << theta << ' ' << "phi = " << phi << ' ' << "pixel_value = " << pixel_value << std::endl;
            }//std::cout << std::endl;
        }
        delete[] sh_value;
        cv::imwrite(store_path, gray);
    }
};

#endif
