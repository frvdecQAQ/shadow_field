#ifndef _FASTMUL_H
#define _FASTMUL_H

#include <fftw3.h>
#include <cstring>
#include "fourierseries.hpp"
#include "select_size.hpp"

constexpr int best_fftsize_above[] = {0,2,2,4,4,6,6,8,8,10,10,12,12,14,14,16,16,20,20,20,20,32,32,32,32,32,32,32,32,32,32,32,32,40,40,40,40,40,40,40,40,48,48,48,48,48,48,48,48,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,72,72,72,72,72,72,72,72,80,80,80,80,80,80,80,80,84,84,84,84,96,96,96,96,96,96,96,96,96,96,96,96,100,100,100,100,112,112,112,112,112,112,112,112,112,112,112,112,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,168,168,168,168,168,168,168,168,192,192,192,192,192,192,192,192,192,192,192,192,192,192,192,192,192,192,192,192,192,192,192,192,200,200,200,200,200,200,200,200,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,280,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,320,324,324,324,324,336,336,336,336,336,336,336,336,336,336,336,336,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,360,384,384,384,384,384,384,384,384,384,384,384,384,384,384,384,384,384,384,384,384,384,384,384,384,392,392,392,392,392,392,392,392,396,396,396,396,399,399,399};


constexpr int bestfftsize(int N) {
	return (N < 400)? best_fftsize_above[N]: N;
}

static int select_size(int n)
{
	// best size >= 4n-3
	if (n==3) return 9;
	if (n==5) return 18;
	if (n==6) return 32;
	if (n==7) return 32;
	if (n==11) return 42;
	return 4*n;
}
// fast multiplication of two 2D Fourier Series using 2D FFT
// (essentially same as multiplication of polynomials)
template <int n>
FourierSeries<2*n-1> fastmul (FourierSeries<n> a, FourierSeries<n> b)
{
	// startup initialization
	static const int N = select_size(n);
	static const int M = 2*n-1;
	static fftwf_complex* const pool0 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
	static fftwf_complex* const poolT = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
	static fftwf_complex* const pool1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
	static fftwf_complex* const pool2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
	static fftwf_plan p1_1 = fftwf_plan_many_dft(1, &N, M, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
	static fftwf_plan p1_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool1, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
	static fftwf_plan p2_1 = fftwf_plan_many_dft(1, &N, M, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
	static fftwf_plan p2_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool2, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
	static fftwf_plan p3_1 = fftwf_plan_many_dft(1, &N, N, pool1, NULL, 1, N, pool2, NULL, 1, N, FFTW_BACKWARD, FFTW_MEASURE);
	// in result we only need column [n-1, 2n-1)
	static fftwf_plan p3_2 = fftwf_plan_many_dft(1, &N, n, pool2+n-1, NULL, N, 1, pool2+n-1, NULL, N, 1, FFTW_BACKWARD, FFTW_MEASURE);
	static void* init0once = memset(pool0, 0, N*N*sizeof(fftwf_complex));
	static void* initTonce = memset(poolT, 0, N*N*sizeof(fftwf_complex));
	// DFT on coefficients of a
	for (int i=0; i<2*n-1; ++i)
		memcpy(pool0+i*N, a.a[i], (2*n-1)*sizeof(fftwf_complex));
	fftwf_execute(p1_1);
	fftwf_execute(p1_2);
	// DFT on coefficients of b, multiplied by coefficient 1/N^2
	for (int i=0; i<2*n-1; ++i)
		for (int j=0; j<2*n-1; ++j)
			*reinterpret_cast<complex*>(pool0+i*N+j) = 1.0f/(N*N) * b.a[i][j];
	fftwf_execute(p2_1);
	fftwf_execute(p2_2);
	// do multiply
	for (int i=0; i<N*N; ++i)
		*reinterpret_cast<complex*>(pool1+i) *= *reinterpret_cast<complex*>(pool2+i);
	// IDFT
	fftwf_execute(p3_1);
	fftwf_execute(p3_2);
	// extract final values
	FourierSeries<2*n-1> c;
	for (int i=0; i<4*n-3; ++i)
		memcpy(c.a[i]+n-1, pool2+i*N+n-1, n*sizeof(fftwf_complex));
	return c;
}

template <int n>
FourierSeries<3*n-2> fastmul (FourierSeries<n> a, FourierSeries<n> b, FourierSeries<n> c)
{
	static const int N = 6 * n;
	static const int M = 2 * n - 1;
	static fftwf_complex* const pool0 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const poolT = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const pool1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const pool2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
       	static fftwf_complex* const pool3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
	//DFT
	static fftwf_plan p1_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p1_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool1, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p2_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p2_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool2, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p3_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
	static fftwf_plan p3_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool3, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
	
	//IDFT
	static fftwf_plan pi_1 = fftwf_plan_many_dft(1, &N, N, pool1, NULL, 1, N, pool2, NULL, 1, N, FFTW_BACKWARD, FFTW_MEASURE);
        static fftwf_plan pi_2 = fftwf_plan_many_dft(1, &N, N, pool2, NULL, N, 1, pool2, NULL, N, 1, FFTW_BACKWARD, FFTW_MEASURE);
        
	static void* init0once = memset(pool0, 0, N*N*sizeof(fftwf_complex));
        static void* initTonce = memset(poolT, 0, N*N*sizeof(fftwf_complex));

	// DFT on coefficients of a
        for (int i=0; i<M; ++i)
                memcpy(pool0+i*N, a.a[i], M*sizeof(fftwf_complex));
        fftwf_execute(p1_1);
        fftwf_execute(p1_2);

	// DFT on coefficients of b, multiplied by coefficient 1/N^2
        for (int i=0; i<M; ++i)
                for (int j=0; j<M; ++j)
                        *reinterpret_cast<complex*>(pool0+i*N+j) = 1.0f/(N*N) * b.a[i][j];
        fftwf_execute(p2_1);
        fftwf_execute(p2_2);

	// DFT on coefficients of b, multiplied by coefficient 1/N^2
        for (int i=0; i<M; ++i)
                for (int j=0; j<M; ++j)
                        *reinterpret_cast<complex*>(pool0+i*N+j) = c.a[i][j];
        fftwf_execute(p3_1);
        fftwf_execute(p3_2);

	// do multiply
        for (int i=0; i<N*N; ++i)
                *reinterpret_cast<complex*>(pool1+i) *= (*reinterpret_cast<complex*>(pool2+i) * *reinterpret_cast<complex*>(pool3+i));

	// IDFT
        fftwf_execute(pi_1);
        fftwf_execute(pi_2);
        // extract final values
        FourierSeries<3*n-2> res;
        for (int i=0; i<6*n-5; ++i)
                memcpy(res.a[i], pool2+i*N, (6*n-5)*sizeof(fftwf_complex));

	return res;
}

template <int n>
FourierSeries<4*n-3> fastmul (FourierSeries<n> a, FourierSeries<n> b, FourierSeries<n> c, FourierSeries<n> d)
{
        static const int N = 8 * n;
        static const int M = 2 * n - 1;
        static fftwf_complex* const pool0 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const poolT = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const pool1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const pool2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const pool3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
	static fftwf_complex* const pool4 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);

        //DFT
        static fftwf_plan p1_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p1_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool1, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p2_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p2_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool2, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p3_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p3_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool3, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
	static fftwf_plan p4_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p4_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool4, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);

        //IDFT
        static fftwf_plan pi_1 = fftwf_plan_many_dft(1, &N, N, pool1, NULL, 1, N, pool2, NULL, 1, N, FFTW_BACKWARD, FFTW_MEASURE);
        static fftwf_plan pi_2 = fftwf_plan_many_dft(1, &N, N, pool2, NULL, N, 1, pool2, NULL, N, 1, FFTW_BACKWARD, FFTW_MEASURE);

        static void* init0once = memset(pool0, 0, N*N*sizeof(fftwf_complex));
        static void* initTonce = memset(poolT, 0, N*N*sizeof(fftwf_complex));

        // DFT on coefficients of a
        for (int i=0; i<M; ++i)
                memcpy(pool0+i*N, a.a[i], M*sizeof(fftwf_complex));
        fftwf_execute(p1_1);
        fftwf_execute(p1_2);

        // DFT on coefficients of b, multiplied by coefficient 1/N^2
        for (int i=0; i<M; ++i)
                for (int j=0; j<M; ++j)
                        *reinterpret_cast<complex*>(pool0+i*N+j) = 1.0f/(N*N) * b.a[i][j];
        fftwf_execute(p2_1);
        fftwf_execute(p2_2);

        // DFT on coefficients of c
        for (int i=0; i<M; ++i)
		memcpy(pool0+i*N, c.a[i], M*sizeof(fftwf_complex));
        fftwf_execute(p3_1);
        fftwf_execute(p3_2);

	// DFT on coefficients of d
	for (int i=0; i<M; ++i)
                memcpy(pool0+i*N, d.a[i], M*sizeof(fftwf_complex));
        fftwf_execute(p4_1);
        fftwf_execute(p4_2);
	
        // do multiply
        for (int i=0; i<N*N; ++i)
                *reinterpret_cast<complex*>(pool1+i) *= (*reinterpret_cast<complex*>(pool2+i) * *reinterpret_cast<complex*>(pool3+i) * *reinterpret_cast<complex*>(pool4+i));

        // IDFT
        fftwf_execute(pi_1);
        fftwf_execute(pi_2);
        // extract final values
        FourierSeries<4*n-3> res;
        for (int i=0; i<8*n-7; ++i)
                memcpy(res.a[i], pool2+i*N, (8*n-7)*sizeof(fftwf_complex));

        return res;
}

template <int n>
FourierSeries<5*n-4> fastmul (FourierSeries<n> a, FourierSeries<n> b, FourierSeries<n> c, FourierSeries<n> d, FourierSeries<n> e)
{
        static const int N = bestfftsize(10 * n - 9);
        static const int M = 2 * n - 1;
        static fftwf_complex* const pool0 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const poolT = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const pool1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const pool2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const pool3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
        static fftwf_complex* const pool4 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);
	static fftwf_complex* const pool5 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N*N);

        //DFT
        static fftwf_plan p1_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p1_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool1, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p2_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p2_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool2, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p3_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p3_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool3, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p4_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p4_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool4, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);
	static fftwf_plan p5_1 = fftwf_plan_many_dft(1, &N, N, pool0, NULL, 1, N, poolT, NULL, 1, N, FFTW_FORWARD, FFTW_MEASURE);
        static fftwf_plan p5_2 = fftwf_plan_many_dft(1, &N, N, poolT, NULL, N, 1, pool5, NULL, N, 1, FFTW_FORWARD, FFTW_MEASURE);


        //IDFT
        static fftwf_plan pi_1 = fftwf_plan_many_dft(1, &N, N, pool1, NULL, 1, N, pool2, NULL, 1, N, FFTW_BACKWARD, FFTW_MEASURE);
        static fftwf_plan pi_2 = fftwf_plan_many_dft(1, &N, N, pool2, NULL, N, 1, pool2, NULL, N, 1, FFTW_BACKWARD, FFTW_MEASURE);

        static void* init0once = memset(pool0, 0, N*N*sizeof(fftwf_complex));
        static void* initTonce = memset(poolT, 0, N*N*sizeof(fftwf_complex));

        // DFT on coefficients of a
        for (int i=0; i<M; ++i)
                memcpy(pool0+i*N, a.a[i], M*sizeof(fftwf_complex));
        fftwf_execute(p1_1);
        fftwf_execute(p1_2);

        // DFT on coefficients of b, multiplied by coefficient 1/N^2
        for (int i=0; i<M; ++i)
                for (int j=0; j<M; ++j)
                        *reinterpret_cast<complex*>(pool0+i*N+j) = 1.0f/(N*N) * b.a[i][j];
        fftwf_execute(p2_1);
        fftwf_execute(p2_2);

        // DFT on coefficients of c
        for (int i=0; i<M; ++i)
                memcpy(pool0+i*N, c.a[i], M*sizeof(fftwf_complex));
        fftwf_execute(p3_1);
        fftwf_execute(p3_2);

        // DFT on coefficients of d
        for (int i=0; i<M; ++i)
                memcpy(pool0+i*N, d.a[i], M*sizeof(fftwf_complex));
        fftwf_execute(p4_1);
        fftwf_execute(p4_2);

	// DFT on coefficients of e
        for (int i=0; i<M; ++i)
                memcpy(pool0+i*N, e.a[i], M*sizeof(fftwf_complex));
        fftwf_execute(p5_1);
        fftwf_execute(p5_2);

        // do multiply
        for (int i=0; i<N*N; ++i)
                *reinterpret_cast<complex*>(pool1+i) *= (*reinterpret_cast<complex*>(pool2+i) * *reinterpret_cast<complex*>(pool3+i) * *reinterpret_cast<complex*>(pool4+i) * *reinterpret_cast<complex*>(pool5+i));

        // IDFT
        fftwf_execute(pi_1);
        fftwf_execute(pi_2);
        // extract final values
        FourierSeries<5*n-4> res;
        for (int i=0; i<10*n-9; ++i)
                memcpy(res.a[i], pool2+i*N, (10*n-9)*sizeof(fftwf_complex));

        return res;
}

#endif

