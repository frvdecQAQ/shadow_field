#include "shRotate.h"


shRotate::shRotate(int band) {
    pre_cos = new float[band + 1];
    pre_sin = new float[band + 1];
    int band2 = band * band;
    coef_tmp = new float[band2];
}

shRotate::~shRotate() {
    delete[] pre_cos;
    delete[] pre_sin;
    delete[] coef_tmp;
}

void shRotate::rotate(const float* coef_in,
	float* coef_out,
	const glm::mat4& m,
	int band) {
	float alpha, beta, gamma;
    --band;
	toZYZ(m, alpha, beta, gamma);
    SHRotateZ(coef_in, coef_out, gamma, band);
    SHRotateXPlus(coef_out, coef_tmp, band);
    SHRotateZ(coef_tmp, coef_out, beta, band);
    SHRotateXMinus(coef_out, coef_tmp, band);
    SHRotateZ(coef_tmp, coef_out, alpha, band);
}

void shRotate::toZYZ(const glm::mat4& m, float& alpha, float& beta, float& gamma) {
    float sy = sqrtf(m[2][1] * m[2][1] + m[2][0] * m[2][0]);
    if (sy > eps) {
        gamma = -atan2f(m[1][2], -m[0][2]);
        beta = -atan2f(sy, m[2][2]);
        alpha = -atan2f(m[2][1], m[2][0]);
    }
    else {
        gamma = 0;
        beta = -atan2f(sy, m[2][2]);
        alpha = -atan2f(-m[1][0], m[1][1]);
    }
}

void shRotate::sinCosIndexed(float s, float c, int n) {
    // s = sin(alpha)
    // c = cos(alpha)
    float si = 0;
    float ci = 1;
    for (int i = 0; i < n; ++i) {
        pre_sin[i] = si;
        pre_cos[i] = ci;
        float old_si = si;
        si = si * c + ci * s;
        ci = ci * c - s * old_si;
    }
}

int shRotate::SHIndex(int l, int m) {
    return l * (l + 1) + m;
}

void shRotate::SHRotateZ(const float* coef_in, float* coef_out, float alpha, int lmax) {
    assert(coef_in != coef_out);
    coef_out[0] = coef_in[0];
    if (lmax == 0)return;
    sinCosIndexed(sinf(alpha), cosf(alpha), lmax + 1);
    // Precompute sine and cosine terms for $z$-axis SH rotation
    for (int l = 1; l <= lmax; ++l) {
        int now_index;
        for (int m = -l; m < 0; ++m) {
            now_index = SHIndex(l, m);
            coef_out[now_index] = pre_cos[-m] * coef_in[now_index] - pre_sin[-m] * coef_in[SHIndex(l, -m)];
        }
        now_index = SHIndex(l, 0);
        coef_out[now_index] = coef_in[now_index];
        for (int m = 1; m <= l; ++m) {
            now_index = SHIndex(l, m);
            coef_out[now_index] = pre_cos[m] * coef_in[now_index] + pre_sin[m] * coef_in[SHIndex(l, -m)];
        }
    }
}

void shRotate::SHRotateXMinus(const float* c_in, float* c_out, int lmax) {
    SHRotateXPlus(c_in, c_out, lmax);

    for (int l = 1; l <= lmax; ++l) {
        float s = (l & 0x1) ? -1.f : 1.f;
        c_out[SHIndex(l, 0)] *= s;
        for (int m = 1; m <= l; ++m) {
            s = -s;
            c_out[SHIndex(l, m)] *= s;
            c_out[SHIndex(l, -m)] *= -s;
        }
    }
}

void shRotate::SHRotateXPlus(const float* c_in, float* c_out, int lmax) {
#define O(l, m)  c_in[SHIndex(l, m)]
    
    *c_out++ = c_in[0];

    if (lmax < 1) return;
    *c_out++ = (O(1, 0));
    *c_out++ = (-1. * O(1, -1));
    *c_out++ = (O(1, 1));

    if (lmax < 2) return;
    *c_out++ = (O(2, 1));
    *c_out++ = (-1. * O(2, -1));
    *c_out++ = (-0.5 * O(2, 0) - 0.8660254037844386 * O(2, 2));
    *c_out++ = (-1. * O(2, -2));
    *c_out++ = (-0.8660254037844386 * O(2, 0) + 0.5 * O(2, 2));

    if (lmax < 3) return;
    *c_out++ = (-0.7905694150420949 * O(3, 0) + 0.6123724356957945 * O(3, 2));
    *c_out++ = (-1. * O(3, -2));
    *c_out++ = (-0.6123724356957945 * O(3, 0) - 0.7905694150420949 * O(3, 2));
    *c_out++ = (0.7905694150420949 * O(3, -3) + 0.6123724356957945 * O(3, -1));
    *c_out++ = (-0.25 * O(3, 1) - 0.9682458365518543 * O(3, 3));
    *c_out++ = (-0.6123724356957945 * O(3, -3) + 0.7905694150420949 * O(3, -1));
    *c_out++ = (-0.9682458365518543 * O(3, 1) + 0.25 * O(3, 3));

    if (lmax < 4) return;
    *c_out++ = (-0.9354143466934853 * O(4, 1) + 0.35355339059327373 * O(4, 3));
    *c_out++ = (-0.75 * O(4, -3) + 0.6614378277661477 * O(4, -1));
    *c_out++ = (-0.35355339059327373 * O(4, 1) - 0.9354143466934853 * O(4, 3));
    *c_out++ = (0.6614378277661477 * O(4, -3) + 0.75 * O(4, -1));
    *c_out++ = (0.375 * O(4, 0) + 0.5590169943749475 * O(4, 2) + 0.739509972887452 * O(4, 4));
    *c_out++ = (0.9354143466934853 * O(4, -4) + 0.35355339059327373 * O(4, -2));
    *c_out++ = (0.5590169943749475 * O(4, 0) + 0.5 * O(4, 2) - 0.6614378277661477 * O(4, 4));
    *c_out++ = (-0.35355339059327373 * O(4, -4) + 0.9354143466934853 * O(4, -2));
    *c_out++ = (0.739509972887452 * O(4, 0) - 0.6614378277661477 * O(4, 2) + 0.125 * O(4, 4));

    if (lmax < 5) return;
    *c_out++ = (0.701560760020114 * O(5, 0) - 0.6846531968814576 * O(5, 2) +
        0.19764235376052372 * O(5, 4));
    *c_out++ = (-0.5 * O(5, -4) + 0.8660254037844386 * O(5, -2));
    *c_out++ = (0.5229125165837972 * O(5, 0) + 0.30618621784789724 * O(5, 2) -
        0.795495128834866 * O(5, 4));
    *c_out++ = (0.8660254037844386 * O(5, -4) + 0.5 * O(5, -2));
    *c_out++ = (0.4841229182759271 * O(5, 0) + 0.6614378277661477 * O(5, 2) +
        0.57282196186948 * O(5, 4));
    *c_out++ = (-0.701560760020114 * O(5, -5) - 0.5229125165837972 * O(5, -3) -
        0.4841229182759271 * O(5, -1));
    *c_out++ = (0.125 * O(5, 1) + 0.4050462936504913 * O(5, 3) + 0.9057110466368399 * O(5, 5));
    *c_out++ = (0.6846531968814576 * O(5, -5) - 0.30618621784789724 * O(5, -3) -
        0.6614378277661477 * O(5, -1));
    *c_out++ = (0.4050462936504913 * O(5, 1) + 0.8125 * O(5, 3) - 0.4192627457812106 * O(5, 5));
    *c_out++ = (-0.19764235376052372 * O(5, -5) + 0.795495128834866 * O(5, -3) -
        0.57282196186948 * O(5, -1));
    *c_out++ = (0.9057110466368399 * O(5, 1) - 0.4192627457812106 * O(5, 3) + 0.0625 * O(5, 5));

    if (lmax < 6) return;
    *c_out++ = (0.879452954966893 * O(6, 1) - 0.46351240544347894 * O(6, 3) +
        0.10825317547305482 * O(6, 5));
    *c_out++ = (-0.3125 * O(6, -5) + 0.8028270361665706 * O(6, -3) - 0.5077524002897476 * O(6, -1));
    *c_out++ = (0.4330127018922193 * O(6, 1) + 0.6846531968814576 * O(6, 3) -
        0.5863019699779287 * O(6, 5));
    *c_out++ = (0.8028270361665706 * O(6, -5) - 0.0625 * O(6, -3) - 0.5929270612815711 * O(6, -1));
    *c_out++ = (0.19764235376052372 * O(6, 1) + 0.5625 * O(6, 3) + 0.8028270361665706 * O(6, 5));
    *c_out++ = (-0.5077524002897476 * O(6, -5) - 0.5929270612815711 * O(6, -3) -
        0.625 * O(6, -1));
    *c_out++ = (-0.3125 * O(6, 0) - 0.45285552331841994 * O(6, 2) - 0.49607837082461076 * O(6, 4) -
        0.6716932893813962 * O(6, 6));
    *c_out++ = (-0.879452954966893 * O(6, -6) - 0.4330127018922193 * O(6, -4) -
        0.19764235376052372 * O(6, -2));
    *c_out++ = (-0.45285552331841994 * O(6, 0) - 0.53125 * O(6, 2) - 0.1711632992203644 * O(6, 4) +
        0.6952686081652184 * O(6, 6));
    *c_out++ = (0.46351240544347894 * O(6, -6) - 0.6846531968814576 * O(6, -4) -
        0.5625 * O(6, -2));
    *c_out++ = (-0.49607837082461076 * O(6, 0) - 0.1711632992203644 * O(6, 2) +
        0.8125 * O(6, 4) - 0.2538762001448738 * O(6, 6));
    *c_out++ = (-0.10825317547305482 * O(6, -6) + 0.5863019699779287 * O(6, -4) -
        0.8028270361665706 * O(6, -2));
    *c_out++ = (-0.6716932893813962 * O(6, 0) + 0.6952686081652184 * O(6, 2) -
        0.2538762001448738 * O(6, 4) + 0.03125 * O(6, 6));

    if (lmax < 7) return;
    *c_out++ = (-0.6472598492877494 * O(7, 0) + 0.6991205412874092 * O(7, 2) -
        0.2981060004427955 * O(7, 4) + 0.05846339666834283 * O(7, 6));
    *c_out++ = (-0.1875 * O(7, -6) + 0.6373774391990981 * O(7, -4) - 0.7473912964438374 * O(7, -2));
    *c_out++ = (-0.47495887979908324 * O(7, 0) - 0.07328774624724109 * O(7, 2) +
        0.78125 * O(7, 4) - 0.3983608994994363 * O(7, 6));
    *c_out++ = (0.6373774391990981 * O(7, -6) - 0.5 * O(7, -4) - 0.5863019699779287 * O(7, -2));
    *c_out++ = (-0.42961647140211 * O(7, 0) - 0.41984465132951254 * O(7, 2) +
        0.10364452469860624 * O(7, 4) + 0.7927281808728639 * O(7, 6));
    *c_out++ = (-0.7473912964438374 * O(7, -6) - 0.5863019699779287 * O(7, -4) -
        0.3125 * O(7, -2));
    *c_out++ = (-0.41339864235384227 * O(7, 0) - 0.5740991584648073 * O(7, 2) -
        0.5385527481129402 * O(7, 4) - 0.4576818286211503 * O(7, 6));
    *c_out++ = (0.6472598492877494 * O(7, -7) + 0.47495887979908324 * O(7, -5) +
        0.42961647140211 * O(7, -3) + 0.41339864235384227 * O(7, -1));
    *c_out++ = (-0.078125 * O(7, 1) - 0.24356964481437335 * O(7, 3) - 0.4487939567607835 * O(7, 5) -
        0.8562442974262661 * O(7, 7));
    *c_out++ = (-0.6991205412874092 * O(7, -7) + 0.07328774624724109 * O(7, -5) +
        0.41984465132951254 * O(7, -3) + 0.5740991584648073 * O(7, -1));
    *c_out++ = (-0.24356964481437335 * O(7, 1) - 0.609375 * O(7, 3) - 0.5700448858423344 * O(7, 5) +
        0.4943528756111367 * O(7, 7));
    *c_out++ = (0.2981060004427955 * O(7, -7) - 0.78125 * O(7, -5) - 0.10364452469860624 * O(7, -3) +
        0.5385527481129402 * O(7, -1));
    *c_out++ = (-0.4487939567607835 * O(7, 1) - 0.5700448858423344 * O(7, 3) + 0.671875 * O(7, 5) -
        0.14905300022139775 * O(7, 7));
    *c_out++ = (-0.05846339666834283 * O(7, -7) + 0.3983608994994363 * O(7, -5) -
        0.7927281808728639 * O(7, -3) + 0.4576818286211503 * O(7, -1));
    *c_out++ = (-0.8562442974262661 * O(7, 1) + 0.4943528756111367 * O(7, 3) -
        0.14905300022139775 * O(7, 5) + 0.015625 * O(7, 7));

    if (lmax < 8) return;
    *c_out++ = (-0.8356088723200586 * O(8, 1) + 0.516334738808072 * O(8, 3) -
        0.184877493221863 * O(8, 5) + 0.03125 * O(8, 7));
    *c_out++ = (-0.109375 * O(8, -7) + 0.4621937330546575 * O(8, -5) - 0.774502108212108 * O(8, -3) +
        0.4178044361600293 * O(8, -1));
    *c_out++ = (-0.4576818286211503 * O(8, 1) - 0.47134697278119864 * O(8, 3) +
        0.7088310138883598 * O(8, 5) - 0.2567449488305466 * O(8, 7));
    *c_out++ = (0.4621937330546575 * O(8, -7) - 0.703125 * O(8, -5) - 0.2181912506838897 * O(8, -3) +
        0.4943528756111367 * O(8, -1));
    *c_out++ = (-0.27421763710600383 * O(8, 1) - 0.6051536478449089 * O(8, 3) -
        0.33802043207474897 * O(8, 5) + 0.6665852814906732 * O(8, 7));
    *c_out++ = (-0.774502108212108 * O(8, -7) - 0.2181912506838897 * O(8, -5) +
        0.265625 * O(8, -3) + 0.5310201708739509 * O(8, -1));
    *c_out++ = (-0.1307281291459493 * O(8, 1) - 0.38081430021731066 * O(8, 3) -
        0.5908647000371574 * O(8, 5) - 0.6991205412874092 * O(8, 7));
    *c_out++ = (0.4178044361600293 * O(8, -7) + 0.4943528756111367 * O(8, -5) +
        0.5310201708739509 * O(8, -3) + 0.546875 * O(8, -1));
    *c_out++ = (0.2734375 * O(8, 0) + 0.3921843874378479 * O(8, 2) + 0.4113264556590057 * O(8, 4) +
        0.4576818286211503 * O(8, 6) + 0.626706654240044 * O(8, 8));
    *c_out++ = (0.8356088723200586 * O(8, -8) + 0.4576818286211503 * O(8, -6) +
        0.27421763710600383 * O(8, -4) + 0.1307281291459493 * O(8, -2));
    *c_out++ = (0.3921843874378479 * O(8, 0) + 0.5 * O(8, 2) + 0.32775276505317236 * O(8, 4) -
        0.6991205412874092 * O(8, 8));
    *c_out++ = (-0.516334738808072 * O(8, -8) + 0.47134697278119864 * O(8, -6) +
        0.6051536478449089 * O(8, -4) + 0.38081430021731066 * O(8, -2));
    *c_out++ = (0.4113264556590057 * O(8, 0) + 0.32775276505317236 * O(8, 2) -
        0.28125 * O(8, 4) - 0.7302075903467452 * O(8, 6) + 0.3332926407453366 * O(8, 8));
    *c_out++ = (0.184877493221863 * O(8, -8) - 0.7088310138883598 * O(8, -6) +
        0.33802043207474897 * O(8, -4) + 0.5908647000371574 * O(8, -2));
    *c_out++ = (0.4576818286211503 * O(8, 0) - 0.7302075903467452 * O(8, 4) + 0.5 * O(8, 6) -
        0.0855816496101822 * O(8, 8));
    *c_out++ = (-0.03125 * O(8, -8) + 0.2567449488305466 * O(8, -6) - 0.6665852814906732 * O(8, -4) +
        0.6991205412874092 * O(8, -2));
    *c_out++ = (0.626706654240044 * O(8, 0) - 0.6991205412874092 * O(8, 2) +
        0.3332926407453366 * O(8, 4) - 0.0855816496101822 * O(8, 6) + 0.0078125 * O(8, 8));

    if (lmax < 9) return;
    *c_out++ = (0.6090493921755238 * O(9, 0) - 0.6968469725305549 * O(9, 2) +
        0.3615761395439417 * O(9, 4) - 0.11158481919598204 * O(9, 6) + 0.016572815184059706 * O(9, 8));
    *c_out++ = (-0.0625 * O(9, -8) + 0.3156095293238149 * O(9, -6) - 0.6817945071647321 * O(9, -4) +
        0.656993626300895 * O(9, -2));
    *c_out++ = (0.44314852502786806 * O(9, 0) - 0.05633673867912483 * O(9, 2) - 0.6723290616859425 * O(9, 4) +
        0.5683291712335379 * O(9, 6) - 0.1594400908746762 * O(9, 8));
    *c_out++ = (0.3156095293238149 * O(9, -8) - 0.71875 * O(9, -6) + 0.20252314682524564 * O(9, -4) +
        0.5854685623498499 * O(9, -2));
    *c_out++ = (0.39636409043643195 * O(9, 0) + 0.25194555463432966 * O(9, 2) - 0.3921843874378479 * O(9, 4) -
        0.6051536478449089 * O(9, 6) + 0.509312687906457 * O(9, 8));
    *c_out++ = (-0.6817945071647321 * O(9, -8) + 0.20252314682524564 * O(9, -6) + 0.5625 * O(9, -4) +
        0.4215855488510013 * O(9, -2));
    *c_out++ = (0.3754879637718099 * O(9, 0) + 0.42961647140211 * O(9, 2) + 0.13799626353637262 * O(9, 4) -
        0.2981060004427955 * O(9, 6) - 0.7526807559068452 * O(9, 8));
    *c_out++ = (0.656993626300895 * O(9, -8) + 0.5854685623498499 * O(9, -6) + 0.4215855488510013 * O(9, -4) +
        0.21875 * O(9, -2));
    *c_out++ = (0.36685490255855924 * O(9, 0) + 0.5130142237306876 * O(9, 2) + 0.4943528756111367 * O(9, 4) +
        0.4576818286211503 * O(9, 6) + 0.38519665736315783 * O(9, 8));
    *c_out++ = (-0.6090493921755238 * O(9, -9) - 0.44314852502786806 * O(9, -7) - 0.39636409043643195 * O(9, -5) -
        0.3754879637718099 * O(9, -3) - 0.36685490255855924 * O(9, -1));
    *c_out++ = (0.0546875 * O(9, 1) + 0.16792332234534904 * O(9, 3) + 0.2954323500185787 * O(9, 5) +
        0.4624247721758373 * O(9, 7) + 0.8171255055356398 * O(9, 9));
    *c_out++ = (0.6968469725305549 * O(9, -9) + 0.05633673867912483 * O(9, -7) - 0.25194555463432966 * O(9, -5) -
        0.42961647140211 * O(9, -3) - 0.5130142237306876 * O(9, -1));
    *c_out++ = (0.16792332234534904 * O(9, 1) + 0.453125 * O(9, 3) + 0.577279787559724 * O(9, 5) +
        0.387251054106054 * O(9, 7) - 0.5322256665703469 * O(9, 9));
    *c_out++ = (-0.3615761395439417 * O(9, -9) + 0.6723290616859425 * O(9, -7) + 0.3921843874378479 * O(9, -5) -
        0.13799626353637262 * O(9, -3) - 0.4943528756111367 * O(9, -1));
    *c_out++ = (0.2954323500185787 * O(9, 1) + 0.577279787559724 * O(9, 3) + 0.140625 * O(9, 5) -
        0.7162405240429014 * O(9, 7) + 0.21608307321780204 * O(9, 9));
    *c_out++ = (0.11158481919598204 * O(9, -9) - 0.5683291712335379 * O(9, -7) + 0.6051536478449089 * O(9, -5) +
        0.2981060004427955 * O(9, -3) - 0.4576818286211503 * O(9, -1));
    *c_out++ = (0.4624247721758373 * O(9, 1) + 0.387251054106054 * O(9, 3) - 0.7162405240429014 * O(9, 5) +
        0.34765625 * O(9, 7) - 0.048317644050206957 * O(9, 9));
    *c_out++ = (-0.016572815184059706 * O(9, -9) + 0.1594400908746762 * O(9, -7) - 0.509312687906457 * O(9, -5) +
        0.7526807559068452 * O(9, -3) - 0.38519665736315783 * O(9, -1));
    *c_out++ = (0.8171255055356398 * O(9, 1) - 0.5322256665703469 * O(9, 3) + 0.21608307321780204 * O(9, 5) -
        0.048317644050206957 * O(9, 7) + 0.00390625 * O(9, 9));
    assert(lmax < 10);
#undef O
}
