#ifndef _SELECT_SIZE_H
#define _SELECT_SIZE_H

constexpr int gpu_best_fftsize_above[] = {0,8,8,8,8,8,8,8,8,9,10,11,12,16,16,16,16,20,20,20,20,21,24,24,24,25,28,28,28,32,32,32,32,35,35,35,40,40,40,40,40,42,42,44,44,48,48,48,48,49,56,56,56,56,56,56,56,64,64,64,64,64,64,64,64,66,66,80,80,80,80,80,80,80,80,80,80,80,80,80,80,81,84,84,84,88,88,88,88,90,90,96,96,96,96,96,96,99,99,99,100,105,105,105,105,105,108,108,108,110,110,112,112,120,120,120,120,120,120,120,120,126,126,126,126,126,126,128,128,135,135,135,135,135,135,135,140,140,140,140,140,144,144,144,144,150,150,150,150,150,150,162,162,162,162,162,162,162,162,162,162,162,162,165,165,165,168,168,168,175,175,175,175,175,175,175,176,180,180,180,180,189,189,189,189,189,189,189,189,189,198,198,198,198,198,198,198,198,198,200,200,210,210,210,210,210,210,210,210,210,210,216,216,216,216,216,216,220,220,220,220,225,225,225,225,225,231,231,231,231,231,231,240,240,240,240,240,240,240,240,240,243,243,243,250,250,250,250,250,250,250,252,252,264,264,264,264,264,264,264,264,264,264,264,264,275,275,275,275,275,275,275,275,275,275,275,280,280,280,280,280,288,288,288,288,288,288,288,288,297,297,297,297,297,297,297,297,297,300,300,300,308,308,308,308,308,308,308,308,315,315,315,315,315,315,315,330,330,330,330,330,330,330,330,330,330,330,330,330,330,330,336,336,336,336,336,336,343,343,343,343,343,343,343,350,350,350,350,350,350,350,352,352,360,360,360,360,360,360,360,360,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,385,396,396,396,396,396,396,396,396,396,396,396,398,398,399};

constexpr int gpu_bestfftsize(int N) {
	return (N < 400)? gpu_best_fftsize_above[N]: N;
}

constexpr int gpu_select_size(int n)
{
	return gpu_bestfftsize(10*n-9);
}

#endif