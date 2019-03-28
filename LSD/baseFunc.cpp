#include <math.h>
#include <baseFunc.h>

const double pi = 4.0 * atan(1.0);

double sind(double x) {
	return sin(x / 180.0 * pi);
}

double cosd(double x) {
	return cos(x / 180.0 * pi);
}

double atand(double x) {
	return atan(x / 180.0 * pi);
}

