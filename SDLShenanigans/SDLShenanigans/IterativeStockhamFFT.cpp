#include <complex>
#include <cmath>
#define M_PI 3.14159265358979323846264338327950288

typedef std::complex<double> complex_t;

void fft0(int length, int stride, bool eo, complex_t* x, complex_t* y)
//x is output if eo == 0, y is output if eo == 1
//x is the input/output sequence depending on eo
//y is the work area or output sequence also depending on eo
//(eo is even and odd)
{
	const int m = length / 2;
	const double theta0 = 2 * M_PI / length;

	if (length == 1) { if (eo) for (int q = 0; q < stride; q++) y[q] = x[q]; }
	else
	{
		for (int p = 0; p < m; p++)
		{
			const complex_t wp = complex_t(cos(p * theta0), -sin(p * theta0));
			for (int q = 0; q < stride; q++)
			{
				const complex_t a = x[q + stride * (p + 0)];
				const complex_t b = x[q + stride * (p + m)];
				y[q + stride * (2 * p + 0)] = a + b;
				y[q + stride * (2 * p + 1)] = (a - b) * wp;
			}
		}
		fft0(length / 2, 2 * stride, !eo, y, x);
	}
}

void fft(int length, complex_t* x) // Fourier Transform
{
	complex_t* y = new complex_t[length];
	fft0(length, 1, 0, x, y);
	delete[]y;
	for (int k = 0; k < length; k++) x[k] /= length;
}

void ifft(int length, complex_t* x) //Inverse Fourier Transform
{
	for (int p = 0; p < length; p++) x[p] = conj(x[p]);
	complex_t* y = new complex_t[length];
	fft0(length , 1, 0, x, y);
	delete[] y;
	for (int k = 0; k < length; k++) x[k] = conj(x[k]);
}

//this is a radix-2 fft algorithm that I dont want to just get rid of.